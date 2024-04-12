"""
Function for influence unlearning[1]

[1]: https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/Wfisher.py
"""

import torch
from torch.autograd import grad
from tqdm import tqdm


def apply_perturb(model, grads):
    """Update model params with influence unlearning"""
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param.view(-1).data += grads[curr : curr + length].data
            curr += length

    return model


def sam_grad(model, loss):
    """Obtain gradients w.r.t to model parameters"""

    params = []
    for param in model.parameters():
        params.append(param)

    sample_grad = grad(loss, params)
    sample_grad = [x.reshape(-1) for x in sample_grad]

    return torch.cat(sample_grad)


def get_grad(args, data_loaders, pipeline, vqvae_latent_dict=None):
    """Calculate grdient for a given dataloader"""

    model = pipeline.unet
    device = pipeline.device

    unique_labels = set()
    model.eval()

    if args.dataset == "celeba":
        vqvae = pipeline.vqvae

    pipeline_scheduler = pipeline.scheduler

    loss_fn = torch.nn.MSELoss()
    total_count = 0

    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    total_grad = torch.zeros_like(torch.cat(params)).to(device)

    for batch in tqdm(data_loaders):

        image, label = batch[0], batch[1]

        if args.precompute_stage == "reuse":
            imageid = batch[2]

        image = image.to(device)

        model.zero_grad()

        batch_size = image.shape[0]

        if args.dataset == "imagenette":
            image = vqvae.encode(image).latent_dist.sample()
            image = image * vqvae.config.scaling_factor
            input_ids_f = label_tokenizer(label).to(device)
            encoder_hidden_states_f = text_encoder(input_ids_f)[0]
        elif args.dataset == "celeba":
            if args.precompute_stage is None:
                # Directly encode the images if there's no precomputation
                image = vqvae.encode(image, False)[0]
            elif args.precompute_stage == "reuse":
                # Retrieve the latent representations.
                image = torch.stack(
                    [vqvae_latent_dict[imageid[i]] for i in range(len(image))]
                ).to(device)
            image = image * vqvae.config.scaling_factor

        noise = torch.randn_like(image).to(device)

        # Antithetic sampling of time steps.
        timesteps = torch.randint(
            0,
            pipeline_scheduler.config.num_train_timesteps,
            (len(image) // 2 + 1,),
            device=image.device,
        ).long()
        timesteps = torch.cat(
            [
                timesteps,
                pipeline_scheduler.config.num_train_timesteps - timesteps - 1,
            ],
            dim=0,
        )[: len(image)]

        noisy_images_f = pipeline_scheduler.add_noise(image, noise, timesteps)

        eps_f = model(noisy_images_f, timesteps).sample
        loss = loss_fn(eps_f, noise)
        grad = sam_grad(model, loss) * batch_size
        total_grad += grad

        if args.by_class:
            new_labels = {l.item() for l in torch.unique(label)}
            new_labels -= unique_labels
            unique_labels.update(new_labels)
            total_count += len(new_labels)
        else:

            total_count += batch_size

    return total_count, total_grad


def woodfisher_diff(args, N, data_loaders, pipeline, grads, vqvae_latent_dict=None):
    """
    Calculate the hessian inverse with woodfisher approximation
    equation (2) in https://arxiv.org/pdf/2004.14340.pdf
    """

    device = pipeline.device
    model = pipeline.unet

    model.eval()

    k_vec = torch.clone(grads)
    o_vec = None

    if args.dataset == "celeba":
        vqvae = pipeline.vqvae

    pipeline_scheduler = pipeline.scheduler

    loss_fn = torch.nn.MSELoss()

    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    for idx, batch in enumerate(tqdm(data_loaders)):

        image, label = batch[0], batch[1]

        if args.precompute_stage == "reuse":
            imageid = batch[2]

        image = image.to(device)

        model.zero_grad()

        if args.dataset == "imagenette":
            image = vqvae.encode(image).latent_dist.sample()
            image = image * vqvae.config.scaling_factor
            input_ids_f = label_tokenizer(label).to(device)
            encoder_hidden_states_f = text_encoder(input_ids_f)[0]
        elif args.dataset == "celeba":
            if args.precompute_stage is None:
                # Directly encode the images if there's no precomputation
                image = vqvae.encode(image, False)[0]
            elif args.precompute_stage == "reuse":
                # Retrieve the latent representations.
                image = torch.stack(
                    [vqvae_latent_dict[imageid[i]] for i in range(len(image))]
                ).to(device)
            image = image * vqvae.config.scaling_factor

        noise = torch.randn_like(image).to(device)

        # Antithetic sampling of time steps.
        timesteps = torch.randint(
            0,
            pipeline_scheduler.config.num_train_timesteps,
            (len(image) // 2 + 1,),
            device=image.device,
        ).long()
        timesteps = torch.cat(
            [
                timesteps,
                pipeline_scheduler.config.num_train_timesteps - timesteps - 1,
            ],
            dim=0,
        )[: len(image)]

        noisy_images_f = pipeline_scheduler.add_noise(image, noise, timesteps)

        eps_f = model(noisy_images_f, timesteps).sample
        loss = loss_fn(eps_f, noise)
        sample_grad = sam_grad(model, loss)

        if idx == 0:
            o_vec = torch.clone(sample_grad)
        else:
            tmp = torch.dot(o_vec, sample_grad)
            k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
            o_vec -= (tmp / (N + tmp)) * o_vec

    return k_vec
