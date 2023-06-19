import torch


# calculating the gradient loss as defined by Eq.7 in the paper
def get_gradient_loss(video_frames_dx, video_frames_dy, jif_current,
                      model_F_mapping, model_F_point_atlas,
                      rgb_output, device, resh, resw, number_of_frames):
    xplus1yt_foreground = torch.cat(
        ((jif_current[0, :] + 1) / (resw / 2) - 1, jif_current[1, :] / (resh / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    xyplus1t_foreground = torch.cat(
        ((jif_current[0, :]) / (resw / 2) - 1, (jif_current[1, :] + 1) / (resh / 2) - 1,
         jif_current[2, :] / (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    # precomputed discrete derivative with respect to x,y direction
    rgb_dx_gt = video_frames_dx[jif_current[2, :], :, jif_current[1, :], jif_current[0, :]].squeeze(1).to(device)
    rgb_dy_gt = video_frames_dy[jif_current[2, :], :, jif_current[1, :], jif_current[0, :]].squeeze(1).to(device)

    # uv coordinates for locations with offsets of 1 pixel
    uv_foreground_xyplus1t = model_F_mapping(xyplus1t_foreground)
    uv_foreground_xplus1yt = model_F_mapping(xplus1yt_foreground)

    # The RGB values (from the 2 layers) for locations with offsets of 1 pixel
    rgb_output_xyplus1t = model_F_point_atlas(uv_foreground_xyplus1t)
    rgb_output_xplus1yt = model_F_point_atlas(uv_foreground_xplus1yt)

    # Use reconstructed RGB values for computing derivatives:
    rgb_dx_output = rgb_output_xplus1yt - rgb_output
    rgb_dy_output = rgb_output_xyplus1t - rgb_output
    gradient_loss = torch.mean(
        (rgb_dx_gt - rgb_dx_output).norm(dim=1) ** 2 + (rgb_dy_gt - rgb_dy_output).norm(dim=1) ** 2)
    return gradient_loss

# get rigidity loss as defined in Eq. 9 in the paper
def get_rigidity_loss(jif_foreground, derivative_amount, resh, resw, number_of_frames, model_F_mapping, uv_foreground, device,
                      uv_mapping_scale=1.0, return_all=False):
    # concatenating (x,y-derivative_amount,t) and (x-derivative_amount,y,t) to get xyt_p:
    is_patch = torch.cat((jif_foreground[1, :] - derivative_amount, jif_foreground[1, :])) / (resh / 2) - 1
    js_patch = torch.cat((jif_foreground[0, :], jif_foreground[0, :] - derivative_amount)) / (resw / 2) - 1
    fs_patch = torch.cat((jif_foreground[2, :], jif_foreground[2, :])) / (number_of_frames / 2.0) - 1
    xyt_p = torch.cat((js_patch, is_patch, fs_patch), dim=1).to(device)

    uv_p = model_F_mapping(xyt_p)
    u_p = uv_p[:, 0].view(2, -1)  # u_p[0,:]= u(x,y-derivative_amount,t).  u_p[1,:]= u(x-derivative_amount,y,t)
    v_p = uv_p[:, 1].view(2, -1)  # v_p[0,:]= u(x,y-derivative_amount,t).  v_p[1,:]= v(x-derivative_amount,y,t)

    u_p_d_ = uv_foreground[:, 0].unsqueeze(
        0) - u_p  # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)   u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
    v_p_d_ = uv_foreground[:, 1].unsqueeze(
        0) - v_p  # v_p_d_[0,:]=u(x,y,t)-v(x,y-derivative_amount,t).  v_p_d_[1,:]= u(x,y,t)-v(x-derivative_amount,y,t).

    # to match units: 1 in uv coordinates is res/2 in image space.
    du_dx = u_p_d_[1, :] * resw / 2
    du_dy = u_p_d_[0, :] * resw / 2
    dv_dy = v_p_d_[0, :] * resh / 2
    dv_dx = v_p_d_[1, :] * resh / 2

    jacobians = torch.cat((torch.cat((du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)), dim=2),
                           torch.cat((dv_dx.unsqueeze(-1).unsqueeze(-1), dv_dy.unsqueeze(-1).unsqueeze(-1)),
                                     dim=2)),
                          dim=1)
    jacobians = jacobians / uv_mapping_scale
    jacobians = jacobians / derivative_amount

    # Apply a loss to constrain the Jacobian to be a rotation matrix as much as possible
    JtJ = torch.matmul(jacobians.transpose(1, 2), jacobians)

    a = JtJ[:, 0, 0] + 0.001
    b = JtJ[:, 0, 1]
    c = JtJ[:, 1, 0]
    d = JtJ[:, 1, 1] + 0.001

    JTJinv = torch.zeros_like(jacobians).to(device)
    JTJinv[:, 0, 0] = d
    JTJinv[:, 0, 1] = -b
    JTJinv[:, 1, 0] = -c
    JTJinv[:, 1, 1] = a
    JTJinv = JTJinv / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))

    # See Equation (9) in the paper:
    rigidity_loss = (JtJ ** 2).sum(1).sum(1).sqrt() + (JTJinv ** 2).sum(1).sum(1).sqrt()

    if return_all:
        return rigidity_loss
    else:
        return rigidity_loss.mean()



# Compute optical flow loss (Eq. 11 in the paper) for all pixels without averaging. This is relevant for visualization of the loss.
def get_optical_flow_loss_all(jif_foreground, uv_foreground,
                              resh, resw, number_of_frames, model_F_mapping,
                              optical_flows, optical_flows_mask, uv_mapping_scale, device,
                              alpha=1.0):
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches_all(
        jif_foreground, optical_flows_mask, optical_flows, resh, resw, number_of_frames)
    uv_foreground_forward_should_match = model_F_mapping(xyt_foreground_forward_should_match.to(device))

    errors = (uv_foreground_forward_should_match - uv_foreground).norm(dim=1)
    errors[relevant_batch_indices_forward == False] = 0
    errors = errors * (alpha.squeeze())

    return errors * resh / (2 * uv_mapping_scale)


# Compute optical flow loss (Eq. 11 in the paper)
def get_optical_flow_loss(jif_foreground, uv_foreground, optical_flows_reverse, optical_flows_reverse_mask, resh, resw,
                          number_of_frames, model_F_mapping, optical_flows, optical_flows_mask, uv_mapping_scale,
                          device, use_alpha=False, alpha=1.0):
    # Forward flow:
    uv_foreground_forward_relevant, xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_mask, optical_flows, resh, resw, number_of_frames, True, uv_foreground)
    uv_foreground_forward_should_match = model_F_mapping(xyt_foreground_forward_should_match.to(device))
    loss_flow_next = (uv_foreground_forward_should_match - uv_foreground_forward_relevant).norm(dim=1) * resh / (
                2 * uv_mapping_scale)

    # Backward flow:
    uv_foreground_backward_relevant, xyt_foreground_backward_should_match, relevant_batch_indices_backward = get_corresponding_flow_matches(
        jif_foreground, optical_flows_reverse_mask, optical_flows_reverse, resh, resw, number_of_frames, False, uv_foreground)
    uv_foreground_backward_should_match = model_F_mapping(xyt_foreground_backward_should_match.to(device))
    loss_flow_prev = (uv_foreground_backward_should_match - uv_foreground_backward_relevant).norm(dim=1) * resh / (
                2 * uv_mapping_scale)

    if use_alpha:
        flow_loss = (loss_flow_prev * alpha[relevant_batch_indices_backward].squeeze()).mean() * 0.5 + (
                    loss_flow_next * alpha[relevant_batch_indices_forward].squeeze()).mean() * 0.5
    else:
        flow_loss = (loss_flow_prev).mean() * 0.5 + (loss_flow_next).mean() * 0.5

    return flow_loss


# A helper function for get_optical_flow_loss to return matching points according to the optical flow
def get_corresponding_flow_matches(jif_foreground, optical_flows_mask, optical_flows, resh, resw, number_of_frames,
                                   is_forward, uv_foreground, use_uv=True):
    batch_forward_mask = torch.where(
        optical_flows_mask[jif_foreground[1, :].squeeze(), jif_foreground[0, :].squeeze(),
        jif_foreground[2, :].squeeze(), :])
    forward_frames_amount = 2 ** batch_forward_mask[1]
    relevant_batch_indices = batch_forward_mask[0]
    jif_foreground_forward_relevant = jif_foreground[:, relevant_batch_indices, 0]
    forward_flows_for_loss = optical_flows[jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0], :,
                             jif_foreground_forward_relevant[2], batch_forward_mask[1]]

    if is_forward:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] + forward_frames_amount))
    else:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] - forward_frames_amount))

    xyt_foreground_forward_should_match = torch.stack((jif_foreground_forward_should_match[0] / (resw / 2) - 1,
                                                       jif_foreground_forward_should_match[1] / (resh / 2) - 1,
                                                       jif_foreground_forward_should_match[2] / (
                                                               number_of_frames / 2) - 1)).T
    if use_uv:
        uv_foreground_forward_relevant = uv_foreground[batch_forward_mask[0]]
        return uv_foreground_forward_relevant, xyt_foreground_forward_should_match, relevant_batch_indices
    else:
        return xyt_foreground_forward_should_match, relevant_batch_indices


# A helper function for get_optical_flow_loss_all to return matching points according to the optical flow
def get_corresponding_flow_matches_all(jif_foreground, optical_flows_mask, optical_flows, resh, resw, number_of_frames,
                                        use_uv=True):
    jif_foreground_forward_relevant = jif_foreground

    forward_flows_for_loss = optical_flows[jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0], :,
                             jif_foreground_forward_relevant[2], 0].squeeze()
    forward_flows_for_loss_mask = optical_flows_mask[
        jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0],
        jif_foreground_forward_relevant[2], 0].squeeze()

    jif_foreground_forward_should_match = torch.stack(
        (jif_foreground_forward_relevant[0].squeeze() + forward_flows_for_loss[:, 0],
         jif_foreground_forward_relevant[1].squeeze() + forward_flows_for_loss[:, 1],
         jif_foreground_forward_relevant[2].squeeze() + 1))

    xyt_foreground_forward_should_match = torch.stack((jif_foreground_forward_should_match[0] / (resw / 2) - 1,
                                                       jif_foreground_forward_should_match[1] / (resh / 2) - 1,
                                                       jif_foreground_forward_should_match[2] / (
                                                               number_of_frames / 2) - 1)).T
    if use_uv:
        return xyt_foreground_forward_should_match, forward_flows_for_loss_mask > 0
    else:
        return 0
