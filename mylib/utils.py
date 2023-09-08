import numpy as np
import torchvision
from torchvision import transforms
import torch


def rm_algo(q_k_npa_list):
    num_k = q_k_npa_list.shape[0]
    d_k = q_k_npa_list.shape[1]
    w_k = q_k_npa_list.shape[2]

#   q_k_param, q_k_npa_list, sign_k_npa, quantized_npa_list = symm_perc_static_quantization(k_npa_list, prec_th, bp_w)
    # q_param, q_npa_list, quantized_npa_list = perc_static_quantization(ifm_npa_list, prec_th, bp_a)

    pim_input = np.array([])
    coord_list = []
    cant_merge_count = 0

    for k_idx, my_k in enumerate(q_k_npa_list[:num_k]):
        my_k_flat = my_k.flatten()
        if pim_input.shape[0] == 0:
            pim_input = np.concatenate((pim_input, my_k_flat[my_k_flat != 0]))
            continue

        # Check for the overlapping values with pim_input vector
        occup_vec = np.full(pim_input.shape, False, dtype=bool)
        cant_merge_idx_list = []

        for w_idx, w in enumerate(my_k_flat):
            # skip zero input
            if w == 0:
                continue

            # check which rows are equal to the current weight value
            eqaul_vec = pim_input == w

            # idx vector for those locations where we can merge
            ok_vec = np.bitwise_and(eqaul_vec, np.bitwise_not(occup_vec))

            # if there are no rows that can be merged
            if sum(ok_vec) == 0:
                # add to a separate list to be appended
                cant_merge_idx_list.append(w_idx)
            else:
                # only leave the first true element of the
                first_true_idx = np.argmax(ok_vec)
                ok_vec[:first_true_idx] = False
                ok_vec[first_true_idx+1:] = False

                # add mapping coordinates
                coord = [w_idx, k_idx, first_true_idx]
                coord_list.append(coord)

                # update occupancy vector
                occup_vec = np.bitwise_or(occup_vec, ok_vec)

        # add mapping coordinates for those rows that cannot be merged
        for count, w_idx in enumerate(cant_merge_idx_list):
            coord = [w_idx, k_idx, count + pim_input.size]
            coord_list.append(coord)
        
        cant_merge_count += len(cant_merge_idx_list)
        pim_input = np.concatenate((pim_input, my_k_flat[cant_merge_idx_list]))

    return len(pim_input), cant_merge_count

def load_CIFAR10(batch_size):
    transform_train = transforms.Compose([
        # resises the image so it can be perfect for our student_model.
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
        transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
        # Normalize all the images
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2471, 0.2435, 0.2616))
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #           'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader
