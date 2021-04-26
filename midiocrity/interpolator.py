from torch import lerp

def interpolate(encoder, decoder, x_s, x_t, length = 10, method = 'lerp'):
    
    interpolator = 0
    if method == 'lerp':
        interpolator = torch.lerp
    elif method == 'slerp':
        zi = slerp

    # Encode enpoints
    z_s = encoder(x_s)
    z_t = encoder(x_t)

    # Interpolation by length L
    z = []
    for i in range(length):
        weight = i / length
        zi = interpolator(z_s, z_t, weight)
        z.append(zi)

    # Decode z
    d = []
    for zi in z:
        d.append(decoder(zi))
    
    # d = [D(z0),...,D(zL)]
    return d

# Based on https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(input, end, weight):
    input_norm = input / torch.norm(input, dim=1, keepdim=True)
    end_norm = end / torch.norm(end, dim=1, keepdim=True)

    theta = torch.acos((input_norm*end_norm).sum(1))
    so = torch.sin(theta)
    res = input * (torch.sin((1.0-weight)*theta)/so).unsqueeze(1) + end * (torch.sin(weight*omega)/so).unsqueeze(1)
    return res