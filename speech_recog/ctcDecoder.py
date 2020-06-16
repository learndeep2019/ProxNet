import torch
import numpy as np

def greedy_decoder(ctc_mtx, blank=40):
    """Greedy decoder. returns hightest probability of class labels for each timestep
    Args:
        ctc_mtx: torch.Tensor(seq_len, num_classes)
        blank: blank labels to collapse

    Returns:
        target: torch.Tensor(predict_len,)
    """
    predict = torch.argmax(ctc_mtx, dim=1)
    collapsed_p = [predict[0]]
    for label in predict:
        if label != collapsed_p[-1]:
            collapsed_p.append(label)

    collapsed_p = [i for i in collapsed_p if i != blank]
    return torch.IntTensor(collapsed_p)


def edit_distance(predict_str, target_str):
    '''
    compute the edit distance between two strings.

    Args:
        predict_str (list): list of predicted labels
        target_str (list): list of ground true labels

    Returns:
        err (int): number of operations to make the two inputs equal.  
    '''
    p_len = len(predict_str) + 1
    t_len = len(target_str) + 1
    sub_optim = np.zeros((p_len, t_len))
    for i in range(p_len):
        sub_optim[i, 0] = i
    for j in range(t_len):
        sub_optim[0, j] = j

    for i in range(1, p_len):
        for j in range(1, t_len):
            if predict_str[i-1] == target_str[j-1]:
                sub_optim[i, j] = min(sub_optim[i-1, j] + 1,
                                      sub_optim[i-1, j-1],
                                      sub_optim[i, j-1] + 1)
            else:
                sub_optim[i, j] = min(sub_optim[i-1, j] + 1,
                                      sub_optim[i-1, j-1] + 1,
                                      sub_optim[i, j-1] + 1)

    return sub_optim[p_len-1, t_len-1]

def phone_error(log_prob_mtx, target, target_len, blank=40):
    """compute phone error rate.
    Args:
        log_prob_mtx: torch.Tensor(seq_len, batch_size, num_classes)
        target: torch.Tensor(batch_size, max_seq_len)
        target_len: torch.Tensor(batch_size,)
    Returns:
        totol_phone: totoal phones in the batch of target
        err_phone: total number of edit operations
    """
    total_phone = target_len.sum().item()
    err_phone = 0
    batch_size = log_prob_mtx.shape[1]
    for i in range(batch_size):
        predict_str = greedy_decoder(log_prob_mtx[:,i,:], blank=blank)
        err_phone += edit_distance(predict_str, target[i,:target_len[i]])

    return total_phone, err_phone

if __name__ == "__main__":
    str1 = torch.randint(1,20,(12,))
    str2 = torch.randint(1,20,(15,))
    err1 = edit_distance(str1, str2)
    err2 = edit_distance(str2, str1)
    print(err1, err2)
    # print(err/len(str2))