import torch


def get_horse_shoe_bounds():
    return torch.tensor([[-0.75,0.75],[-0.6,0.75]])


def horse_shoe_sdf(p, angle=torch.pi*.75, ra=.5, width=.1):
    """ SDF of an arc """
    assert 0 < angle < torch.pi, "angle must be between 0 and PI. Don't ask"
    # p is a 2D tensor (torch.Tensor in PyTorch)
    p_ = p.clone()
    p_[:, 0] = torch.abs(p[:, 0])
    # sc is the sin/cos of the arc's aperture, also a 2D tensor
    angle = torch.tensor(angle)
    sc = torch.stack([torch.sin(angle), torch.cos(angle)])
    
    # Check the condition
    condition = sc[1]*p_[:,0] > sc[0]*p_[:,1]
    
    # Calculate distance based on the condition
    sdf = -width + torch.where(condition,
            torch.norm(p_ - sc * ra, dim=-1) , ## SDF of the ring
            torch.abs(torch.norm(p_, dim=-1) - ra) ## SDF of the round tips
        )
    
    return sdf