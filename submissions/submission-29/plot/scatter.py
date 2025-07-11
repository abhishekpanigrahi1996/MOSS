import matplotlib.pyplot as plt
import json


def pair_to_string(pair):
    return "t[{}]a[{}]v[{}]".format(pair["text"], pair["audio"], pair["video"])


def config_to_string(config):
    name = ""
    for pair in config:
        name += pair_to_string(pair) + "_"

    return name[:-1]


def scatter(x, y, path):
    fig, _ = plt.subplots()
    plt.xlabel("MI")
    plt.ylabel("MSE")
    plt.title(path.split("/")[-1])
    plt.scatter(x, y)
    # plt.axhline(y=0.7754535738352426)
    plt.axhline(y=0.7027587890625)
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    mis = [[[{'text': 'bert_cls', 'audio': 'l0', 'video': 'l0'}], 0.04453505181404557],
           [[{'text': 'bert_cls', 'audio': 'l0', 'video': 'l1'}], 0.07049563382122534],
           [[{'text': 'bert_cls', 'audio': 'l1', 'video': 'l0'}], 0.07052346599834122],
           [[{'text': 'bert_cls', 'audio': 'l1', 'video': 'l1'}], 0.20084192640583728],
           [[{'text': 'bert_latent0', 'audio': 'l0', 'video': 'l0'}], 0.01981001005414504],
           [[{'text': 'bert_latent0', 'audio': 'l0', 'video': 'l1'}], 0.02566473435379754],
           [[{'text': 'bert_latent0', 'audio': 'l1', 'video': 'l0'}], 0.03883076328771948],
           [[{'text': 'bert_latent0', 'audio': 'l1', 'video': 'l1'}], 0.1973378876723769],
           [[{'text': 'bert_latent1', 'audio': 'l0', 'video': 'l0'}], 0.020002897066995956],
           [[{'text': 'bert_latent1', 'audio': 'l0', 'video': 'l1'}], 0.027235898922371632],
           [[{'text': 'bert_latent1', 'audio': 'l1', 'video': 'l0'}], 0.038511096687833785],
           [[{'text': 'bert_latent1', 'audio': 'l1', 'video': 'l1'}], 0.19883448307021587],
           [[{'text': 'bert_latent2', 'audio': 'l0', 'video': 'l0'}], 0.01904373361435438],
           [[{'text': 'bert_latent2', 'audio': 'l0', 'video': 'l1'}], 0.04322107363253144],
           [[{'text': 'bert_latent2', 'audio': 'l1', 'video': 'l0'}], 0.03853909691768584],
           [[{'text': 'bert_latent2', 'audio': 'l1', 'video': 'l1'}], 0.20484356865413508],
           [[{'text': 'bert_latent3', 'audio': 'l0', 'video': 'l0'}], 0.01856940385732861], [[{'text': 'bert_latent3', 'audio': 'l0', 'video': 'l1'}], 0.03942335414911588], [[{'text': 'bert_latent3', 'audio': 'l1', 'video': 'l0'}], 0.03855668206346982], [[{'text': 'bert_latent3', 'audio': 'l1', 'video': 'l1'}], 0.20717885014889367], [[{'text': 'bert_latent4', 'audio': 'l0', 'video': 'l0'}], 0.018318942102085373], [[{'text': 'bert_latent4', 'audio': 'l0', 'video': 'l1'}], 0.041557955924916484], [[{'text': 'bert_latent4', 'audio': 'l1', 'video': 'l0'}], 0.039159959078103185], [[{'text': 'bert_latent4', 'audio': 'l1', 'video': 'l1'}], 0.20587822884289272], [[{'text': 'bert_latent5', 'audio': 'l0', 'video': 'l0'}], 0.020980516785614384], [[{'text': 'bert_latent5', 'audio': 'l0', 'video': 'l1'}], 0.06529056238819328], [[{'text': 'bert_latent5', 'audio': 'l1', 'video': 'l0'}], 0.06307952113944756], [[{'text': 'bert_latent5', 'audio': 'l1', 'video': 'l1'}], 0.2002453424085123], [[{'text': 'bert_latent6', 'audio': 'l0', 'video': 'l0'}], 0.050833286783907056], [[{'text': 'bert_latent6', 'audio': 'l0', 'video': 'l1'}], 0.06834885589642545], [[{'text': 'bert_latent6', 'audio': 'l1', 'video': 'l0'}], 0.06781793261548627], [[{'text': 'bert_latent6', 'audio': 'l1', 'video': 'l1'}], 0.19892508487004554], [[{'text': 'bert_latent7', 'audio': 'l0', 'video': 'l0'}], 0.051794867096192604], [[{'text': 'bert_latent7', 'audio': 'l0', 'video': 'l1'}], 0.06839555002294281], [[{'text': 'bert_latent7', 'audio': 'l1', 'video': 'l0'}], 0.06847574271190501], [[{'text': 'bert_latent7', 'audio': 'l1', 'video': 'l1'}], 0.19897752673084787], [[{'text': 'bert_latent8', 'audio': 'l0', 'video': 'l0'}], 0.051788644138704626], [[{'text': 'bert_latent8', 'audio': 'l0', 'video': 'l1'}], 0.06837760037135018], [[{'text': 'bert_latent8', 'audio': 'l1', 'video': 'l0'}], 0.06869547254159455], [[{'text': 'bert_latent8', 'audio': 'l1', 'video': 'l1'}], 0.19948503571630147], [[{'text': 'bert_latent9', 'audio': 'l0', 'video': 'l0'}], 0.05241909763409089], [[{'text': 'bert_latent9', 'audio': 'l0', 'video': 'l1'}], 0.0684674610464254], [[{'text': 'bert_latent9', 'audio': 'l1', 'video': 'l0'}], 0.06900459203972295], [[{'text': 'bert_latent9', 'audio': 'l1', 'video': 'l1'}], 0.19944362561096915], [[{'text': 'bert_latent10', 'audio': 'l0', 'video': 'l0'}], 0.049371076836452005], [[{'text': 'bert_latent10', 'audio': 'l0', 'video': 'l1'}], 0.06470353942793969], [[{'text': 'bert_latent10', 'audio': 'l1', 'video': 'l0'}], 0.06607409463252742], [[{'text': 'bert_latent10', 'audio': 'l1', 'video': 'l1'}], 0.06502483276067869], [[{'text': 'bert_latent11', 'audio': 'l0', 'video': 'l0'}], 0.0484647379251347], [[{'text': 'bert_latent11', 'audio': 'l0', 'video': 'l1'}], 0.06580586661507346], [[{'text': 'bert_latent11', 'audio': 'l1', 'video': 'l0'}], 0.06497879924211189], [[{'text': 'bert_latent11', 'audio': 'l1', 'video': 'l1'}], 0.0581307600805315]]

    mis_club = [[[{'text': 'bert_cls', 'audio': 'l0', 'video': 'l0'}], 0.11141520374942393], [[{'text': 'bert_cls', 'audio': 'l0', 'video': 'l1'}], 0.12691859188415702], [[{'text': 'bert_cls', 'audio': 'l1', 'video': 'l0'}], 0.1424804845795272], [[{'text': 'bert_cls', 'audio': 'l1', 'video': 'l1'}], 0.2689871054201845], [[{'text': 'bert_latent0', 'audio': 'l0', 'video': 'l0'}], 0.041368351214461856], [[{'text': 'bert_latent0', 'audio': 'l0', 'video': 'l1'}], 0.04625239378462235], [[{'text': 'bert_latent0', 'audio': 'l1', 'video': 'l0'}], 0.09425338135204381], [[{'text': 'bert_latent0', 'audio': 'l1', 'video': 'l1'}], 0.23291474758159547], [[{'text': 'bert_latent1', 'audio': 'l0', 'video': 'l0'}], 0.04271189769404748], [[{'text': 'bert_latent1', 'audio': 'l0', 'video': 'l1'}], 0.060223168833920405], [[{'text': 'bert_latent1', 'audio': 'l1', 'video': 'l0'}], 0.08232594120301424], [[{'text': 'bert_latent1', 'audio': 'l1', 'video': 'l1'}], 0.23683571442961693], [[{'text': 'bert_latent2', 'audio': 'l0', 'video': 'l0'}], 0.031003500867102827], [[{'text': 'bert_latent2', 'audio': 'l0', 'video': 'l1'}], 0.0788558323438915], [[{'text': 'bert_latent2', 'audio': 'l1', 'video': 'l0'}], 0.08616654212690061], [[{'text': 'bert_latent2', 'audio': 'l1', 'video': 'l1'}], 0.2498660988750912], [[{'text': 'bert_latent3', 'audio': 'l0', 'video': 'l0'}], 0.034758230575197745], [[{'text': 'bert_latent3', 'audio': 'l0', 'video': 'l1'}], 0.062200108276946206], [[{'text': 'bert_latent3', 'audio': 'l1', 'video': 'l0'}], 0.1021157456561923], [[{'text': 'bert_latent3', 'audio': 'l1', 'video': 'l1'}], 0.2623206003909073], [[{'text': 'bert_latent4', 'audio': 'l0', 'video': 'l0'}], 0.03548016159662178], [[{'text': 'bert_latent4', 'audio': 'l0', 'video': 'l1'}], 0.08339292016471662], [[{'text': 'bert_latent4', 'audio': 'l1', 'video': 'l0'}], 0.09427312288492445], [[{'text': 'bert_latent4', 'audio': 'l1', 'video': 'l1'}], 0.2830109404666083], [[{'text': 'bert_latent5', 'audio': 'l0', 'video': 'l0'}], 0.10325848725106981], [[{'text': 'bert_latent5', 'audio': 'l0', 'video': 'l1'}], 0.11628749558613413], [[{'text': 'bert_latent5', 'audio': 'l1', 'video': 'l0'}], 0.14885678843018554], [[{'text': 'bert_latent5', 'audio': 'l1', 'video': 'l1'}], 0.27331329361786916], [[{'text': 'bert_latent6', 'audio': 'l0', 'video': 'l0'}], 0.11584442182784042], [[{'text': 'bert_latent6', 'audio': 'l0', 'video': 'l1'}], 0.12935305280344828], [[{'text': 'bert_latent6', 'audio': 'l1', 'video': 'l0'}], 0.16079520756408336], [[{'text': 'bert_latent6', 'audio': 'l1', 'video': 'l1'}], 0.26645539109669036], [[{'text': 'bert_latent7', 'audio': 'l0', 'video': 'l0'}], 0.11355283066985153], [[{'text': 'bert_latent7', 'audio': 'l0', 'video': 'l1'}], 0.07853466267919257], [[{'text': 'bert_latent7', 'audio': 'l1', 'video': 'l0'}], 0.15182285551868735], [[{'text': 'bert_latent7', 'audio': 'l1', 'video': 'l1'}], 0.26651596053252147], [[{'text': 'bert_latent8', 'audio': 'l0', 'video': 'l0'}], 0.1259394194043818], [[{'text': 'bert_latent8', 'audio': 'l0', 'video': 'l1'}], 0.12265661666317591], [[{'text': 'bert_latent8', 'audio': 'l1', 'video': 'l0'}], 0.15842923202684947], [[{'text': 'bert_latent8', 'audio': 'l1', 'video': 'l1'}], 0.2600212028575322], [[{'text': 'bert_latent9', 'audio': 'l0', 'video': 'l0'}], 0.11195813364807576], [[{'text': 'bert_latent9', 'audio': 'l0', 'video': 'l1'}], 0.13318036348810272], [[{'text': 'bert_latent9', 'audio': 'l1', 'video': 'l0'}], 0.15718435975057737], [[{'text': 'bert_latent9', 'audio': 'l1', 'video': 'l1'}], 0.2716744667480862], [[{'text': 'bert_latent10', 'audio': 'l0', 'video': 'l0'}], 0.10756933044583078], [[{'text': 'bert_latent10', 'audio': 'l0', 'video': 'l1'}], 0.12324835936583224], [[{'text': 'bert_latent10', 'audio': 'l1', 'video': 'l0'}], 0.14721816482525024], [[{'text': 'bert_latent10', 'audio': 'l1', 'video': 'l1'}], 0.17078232132489718], [[{'text': 'bert_latent11', 'audio': 'l0', 'video': 'l0'}], 0.12189093001541637], [[{'text': 'bert_latent11', 'audio': 'l0', 'video': 'l1'}], 0.1281710242823003], [[{'text': 'bert_latent11', 'audio': 'l1', 'video': 'l0'}], 0.16043745980612814], [[{'text': 'bert_latent11', 'audio': 'l1', 'video': 'l1'}], 0.09779088666278218]]

    performance = json.load(open("../results_l1.json"))
    x = []
    y = []
    for p in mis_club:
        str_key = config_to_string(p[0])
        x.append(p[1])
        y.append(performance[str_key]["test_mse"])

    scatter(x, y, "./l1scatter_club_mse.png")