"""Function for influence unlearning"""

import torch
from torch.autograd import grad
from tqdm import tqdm

from imagenet import get_x_y_from_data_dict

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


def apply_perturb(model, v):
    curr = 0
    for param in model.parameters():
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr : curr + length].data
        curr += length


def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = len(train_dl)

    for idx, (data, label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        if idx == 0:
            o_vec = torch.clone(sample_grad)
        else:
            tmp = torch.dot(o_vec, sample_grad)
            k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
            o_vec -= (tmp / (N + tmp)) * o_vec
    return k_vec


def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr : curr + length].view(param.shape)
            curr += length


def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000
    o_vec = None
    for idx, (data, label) in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec


def woodfisher_im(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 300000
    o_vec = None
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    for idx, data in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data, label = get_x_y_from_data_dict(data, device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec


def get_grad(args, data_loaders, pipeline, vqvae_latent_dict=None):
    """Calculate grdient for a given dataloader"""

    model = pipeline.unet
    model.eval()

    if args.dataset == "celeba":
        vqvae = pipeline.vqvae

    pipeline_scheduler = pipeline.scheduler

    loss_fn = torch.nn.MSELoss()
    total = 0

    for batch in data_loaders:

        image, label = batch[0], batch[1]

        if args.precompute_stage == "reuse":
            imageid = batch[2]

        image = image.to(device)

        model.zero_grad()

        real_num = image.shape[0]

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
                    [vqvae_latent_dict[imageid_f[i]] for i in range(len(image))]
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
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    return total




def Wfisher(data_loaders, model, criterion, args):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=args.batch_size, shuffle=False
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=args.batch_size, shuffle=False
    )
    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    model.eval()
    if args.imagenet_arch:
        for i, data in enumerate(tqdm(forget_loader)):
            model.zero_grad()
            data, label = get_x_y_from_data_dict(data, device)
            real_num = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            f_grad = sam_grad(model, loss) * real_num
            forget_grad += f_grad
            total += real_num
        total_2 = 0
        for i, data in enumerate(tqdm(retain_grad_loader)):
            model.zero_grad()
            data, label = get_x_y_from_data_dict(data, device)
            real_num = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            r_grad = sam_grad(model, loss) * real_num
            retain_grad += r_grad
            total_2 += real_num
    else:
        for i, (data, label) in enumerate(tqdm(forget_loader)):
            model.zero_grad()
            real_num = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            f_grad = sam_grad(model, loss) * real_num
            forget_grad += f_grad
            total += real_num

        total_2 = 0
        for i, (data, label) in enumerate(tqdm(retain_grad_loader)):
            model.zero_grad()
            real_num = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            r_grad = sam_grad(model, loss) * real_num
            retain_grad += r_grad
            total_2 += real_num
    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2
    if args.imagenet_arch:
        perturb = woodfisher_im(
            model,
            retain_loader,
            device=device,
            criterion=criterion,
            v=forget_grad - retain_grad,
        )
    else:
        perturb = woodfisher(
            model,
            retain_loader,
            device=device,
            criterion=criterion,
            v=forget_grad - retain_grad,
        )
    apply_perturb(model, args.alpha * perturb)

    return model