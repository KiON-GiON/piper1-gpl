import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l_dg = torch.mean((1 - dg) ** 2)
        gen_losses.append(l_dg)
        loss += l_dg

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l_kl = kl / torch.sum(z_mask)
    return l_kl

def _mask_like_disc_output(d: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    """
    d: salida del dur-disc. Suele ser [b, t, 1] (si viene de un Linear sobre time)
       o [b, 1, t] (si fuese conv-style).
    x_mask: [b, 1, t]
    devuelve mask broadcastable a d, dtype float, mismo device.
    """
    x_mask = x_mask.float()

    if d.dim() != 3:
        # fallback: sin máscara
        return torch.ones_like(d, dtype=torch.float32)

    # d: [b, t, 1]
    if d.shape[1] == x_mask.shape[2] and d.shape[2] == 1:
        return x_mask.transpose(1, 2)  # [b, t, 1]

    # d: [b, 1, t]
    if d.shape[2] == x_mask.shape[2] and d.shape[1] == 1:
        return x_mask  # [b, 1, t]

    # si no coincide, no aplicamos máscara (mejor que romper)
    return torch.ones_like(d, dtype=torch.float32)


def masked_discriminator_loss(disc_real_outputs, disc_generated_outputs, x_mask: torch.Tensor):
    loss = 0.0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()

        m = _mask_like_disc_output(dr, x_mask).to(device=dr.device, dtype=dr.dtype)
        denom = torch.clamp_min(m.sum(), 1.0)

        r_loss = (((1.0 - dr) ** 2) * m).sum() / denom
        g_loss = ((dg ** 2) * m).sum() / denom

        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def masked_generator_loss(disc_outputs, x_mask: torch.Tensor):
    loss = 0.0
    gen_losses = []

    for dg in disc_outputs:
        dg = dg.float()

        m = _mask_like_disc_output(dg, x_mask).to(device=dg.device, dtype=dg.dtype)
        denom = torch.clamp_min(m.sum(), 1.0)

        l = (((1.0 - dg) ** 2) * m).sum() / denom
        gen_losses.append(l.item())
        loss = loss + l

    return loss, gen_losses
