import pickle
from pathlib import Path

import chess
import pandas as pd
import torch
from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

MODEL_DIR = "models/"
DATA_DIR = "data/"
D_MODEL = 512
N_HEADS = 8
WANDB_LOGGING = False

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

with open(f"{MODEL_DIR}meta.pkl", "rb") as f:
    meta = pickle.load(f)

stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

meta_round_trip_input = "1.e4 e6 2.Nf3"
print(encode(meta_round_trip_input))
print("Performing round trip test on meta")
assert decode(encode(meta_round_trip_input)) == meta_round_trip_input

def get_transformer_lens_model(
    model_name: str, n_layers: int, device: torch.device
) -> HookedTransformer:

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=D_MODEL,
        d_head=int(D_MODEL / N_HEADS),
        n_heads=N_HEADS,
        d_mlp=D_MODEL * 4,
        d_vocab=32,
        n_ctx=1023,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(f"{MODEL_DIR}{model_name}.pth"))
    model.to(device)
    return model

def get_board_seqs_int(df: pd.DataFrame) -> Int[Tensor, "num_games pgn_str_length"]:
    encoded_df = df["transcript"].apply(encode)
    board_seqs_int_Bl = torch.tensor(encoded_df.apply(list).tolist())
    return board_seqs_int_Bl 


def get_board_seqs_string(df: pd.DataFrame) -> list[str]:

    key = "transcript"
    row_length = len(df[key].iloc[0])

    assert all(
        df[key].apply(lambda x: len(x) == row_length)
    ), "Not all transcripts are of length {}".format(row_length)

    board_seqs_string_Bl = df[key]

    return board_seqs_string_Bl    

PIECE_TO_INT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

INT_TO_PIECE = {value: key for key, value in PIECE_TO_INT.items()}

def get_board_state(board: chess.Board) -> torch.Tensor:
    state_RR = torch.zeros((8, 8), dtype=torch.int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_value = PIECE_TO_INT[piece.piece_type]
            # Multiply by -1 if the piece is black
            if piece.color == chess.BLACK:
                piece_value *= -1
            state_RR[i // 8, i % 8] = piece_value
    return state_RR

def filter_moves(moves_string: str):
    # for each move, compute the dot index, the move index, and the actual move string
    cur_dot = None
    cur_move_str = ""

    moves = []

    for i, c in enumerate(moves_string):
        if c == ".":
            cur_dot = i
        elif cur_dot is not None:
            if c == " ":
                # if we encounter a space, we have reached the end of the move
                # we can now extract the move string
                moves.append((cur_dot, i - 1, cur_move_str))

                # reset state
                cur_dot = None
                cur_move_str = ""
            else:
                cur_move_str += c
    if cur_dot is not None:
        # we might still have a move at the end
        moves.append((cur_dot, len(moves_string) - 1, cur_move_str))

    for i in range(len(moves)):
        # deal with check and checkmate
        if moves_string[moves[i][1]] == "+" or moves_string[moves[i][1]] == "#":
            moves[i] = (moves[i][0], moves[i][1] - 1, moves[i][2])

    return moves

def get_dots_indices(moves_string: str):
    indices = [index for index, char in enumerate(moves_string) if char == "."]
    return torch.tensor(indices)

def get_indices(moves_string: str):
    """
    Filter out the indices of the moves: returns the following information for each move:
    - The index of the move
    - The piece that was moved (or the piece that was captured)
    - The board position after the move
    - The to square
    - The from square (-1 if the piece was captured)
    """
                 
    # map the indices to the moves
    indices = []
    piece_types = [] # tuple with capture and piece type
    board_states = []
    to_squares = []
    from_squares = []

    board = chess.Board()

    last_capture = False
    last_piece = None
    last_square = None

    for move in filter_moves(moves_string):
        dot_idx, move_idx, white_move = move

        # check if we need to add the dot index to the move
        if last_capture:
            indices.append(dot_idx)
            piece_types.append((True, last_piece.piece_type))
            board_states.append(get_board_state(board))
            to_squares.append(last_square)
            from_squares.append(-1)
        
        last_capture = False
        last_piece = None
        last_square = None

        piece = None
        try:
            mv = board.parse_san(white_move)
            piece = board.piece_at(mv.from_square)
            board.push_san(white_move)
        except:
            break

        if piece is not None:
            indices.append(move_idx)
            piece_types.append((False, piece.piece_type))
            board_states.append(get_board_state(board))
            to_squares.append(mv.to_square)
            from_squares.append(mv.from_square)
        
        # get the black move
        black_move = moves_string[move_idx + 2 : moves_string.find(" ", move_idx + 2)]

        try:
            mv = board.parse_san(black_move) 
            if board.is_capture(mv):
                last_capture = True
                last_square = mv.to_square
                last_piece = board.piece_at(mv.to_square) # we want to push the piece that was captured
                assert last_piece is not None
            piece = board.piece_at(mv.from_square)
            board.push_san(black_move)
        except:
            break

    # assert that all the lengths are the same
    assert len(indices) == len(piece_types)
    assert len(piece_types) == len(board_states)
    assert len(board_states) == len(to_squares)
    assert len(to_squares) == len(from_squares)

    return torch.tensor(indices), piece_types, torch.stack(board_states), torch.tensor(to_squares), torch.tensor(from_squares)

def prepare_data_head(model, df: pd.DataFrame, attn_layer: int = 5):
    board_seqs_int_Bl = get_board_seqs_int(df)
    board_seqs_str_Bl = get_board_seqs_string(df)

    W_V = model.blocks[attn_layer].attn.W_V # [n_heads, d_model, d_head]

    # move properties
    to_list, from_list, piece_type_list, board_stack_list = [], [], [], []

    # game properties
    index_list, game_index_list = [], []

    # model properties
    head_v_list = []

    # dots properties
    dots_index_list, dots_game_index_list = [], []
    dots_attn_list = []
    # get the actual head contributions
    dots_head_contributions = []
    
    game_count = 0

    hook_ln1 = f"blocks.{attn_layer}.ln1.hook_normalized"
    hook_attn = f"blocks.{attn_layer}.attn.hook_pattern"

    for seqs_int, seq_str in tqdm(zip(board_seqs_int_Bl, board_seqs_str_Bl),
                                  total=len(board_seqs_int_Bl)):

        dots_indices = get_dots_indices(seq_str)
        move_indices, piece_types, board_states, to_sq, from_sq = get_indices(seq_str)

        def ln1_hook(value, hook):
            ln1_act[0] = value.detach()
            return value

        def attn_hook(attn, hook):
            # attn: [batch, heads, seq, seq]
            layer_attn[0] = attn.detach()
            return attn

        ln1_act = [None]
        layer_attn = [None]

        #with model.hooks(fwd_hooks=[(hook_name, ln1_hook), (attn_hook_name, attn_hook)]):
        with model.hooks(fwd_hooks=[(hook_ln1, ln1_hook), (hook_attn, attn_hook)]): # don't care about attention
            model(seqs_int.unsqueeze(0))
        
        ln1 = ln1_act[0]
        v = torch.einsum(
            "b s m, h m d -> b s h d", 
            ln1, W_V
        )
        head_v_list.append(v[:, move_indices, :, :].detach().cpu())
        attn = layer_attn[0]

        head_contributions = torch.einsum(
            "b h s t, b t h d -> b h s d",  
            attn, v
        )

        index_list.append(move_indices)
        game_index_list.append(torch.full_like(move_indices, game_count))

        dots_index_list.append(torch.full_like(dots_indices, game_count))
        dots_game_index_list.append(dots_indices)

        # drop the batch dimension and append and move to cpu
        dots_head_contributions.append(head_contributions[:, :, dots_indices, :].detach().cpu())
        dots_attn_list.append(attn[:, :, dots_indices, :][:, :, :, move_indices].detach().cpu())

        game_count += 1

        board_stack_list.append(board_states)
        to_list.append(to_sq)
        from_list.append(from_sq)

        piece_type_list.append(piece_types)

        del ln1_act, attn
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        

    # combine the data into dictionary
    data = {
        "to": to_list,
        "from": from_list,
        "piece_type": piece_type_list,
        "board_state": board_stack_list,
        "index": index_list,
        "game_index": game_index_list,
        "head_v": head_v_list,
        "dots_index": dots_index_list,
        "dots_game_index": dots_game_index_list,
        "dots_attn": dots_attn_list,
        "dots_head_contributions": dots_head_contributions
    }

    return data


dataset_prefix = "lichess_"
split = "train"
n_layers = 8
model_name = f"tf_lens_{dataset_prefix}{n_layers}layers_ckpt_no_optimizer"

model = get_transformer_lens_model(model_name, n_layers, DEVICE)

first_layer = 0
last_layer = 7
layers = list(range(first_layer, last_layer + 1))

attn_layer = 5 #layer we want to probe

cache_file = f"{DATA_DIR}/precomputed_game_cache.pt"
input_file = f"{DATA_DIR}/lichess_train.csv"

df = pd.read_csv(input_file)
df = df[:10000]
data = prepare_data_head(model, df, attn_layer=attn_layer)

cache_file = Path(cache_file)

# unpack the dict and save
torch.save(
    data,
    cache_file,
)
