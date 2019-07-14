import cPickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def softmax(x):
    max_of_dim1 = np.max(x, axis=0, keepdims=True)
    return (np.exp(x - max_of_dim1).T / np.exp(x - max_of_dim1).sum(axis=0, keepdims=True).T).T
def norm(x):
    x = np.array(x)
    return x * 1.0 / (x[0]+x[1])

def extract(rules):
    # load data id
    print 'loading train id'
    train_id = np.loadtxt(open("./ind/ExFashion/id/train_ijk_shuffled_811.txt"), delimiter="\t",skiprows=0)
    train_id = np.array(train_id)
    train_set_i = np.array(train_id[:, 0])
    train_set_j = np.array(train_id[:, 1])
    train_set_k = np.array(train_id[:, 2])
    print 'loading valid id'
    valid_id = np.loadtxt(open("./ind/ExFashion/id/valid_ijk_shuffled_811.txt"), delimiter="\t",skiprows=0)
    valid_id = np.array(valid_id)
    valid_set_i = np.array(valid_id[:, 0])
    valid_set_j = np.array(valid_id[:, 1])
    valid_set_k = np.array(valid_id[:, 2])
    print 'loading test id'
    test_id = np.loadtxt(open("./ind/ExFashion/id/test_ijk_shuffled_811.txt"), delimiter="\t",skiprows=0)
    test_id = np.array(test_id)
    test_set_i = np.array(test_id[:, 0])
    test_set_j = np.array(test_id[:, 1])
    test_set_k = np.array(test_id[:, 2])

    # extract rule ind
    train_rules_ind = []
    valid_rules_ind = []
    test_rules_ind = []
    for rule in rules:
        train_rule_ind = extract_ind(train_set_i, train_set_j, train_set_k, rule)
        valid_rule_ind = extract_ind(valid_set_i, valid_set_j, valid_set_k, rule)
        test_rule_ind = extract_ind(test_set_i, test_set_j, test_set_k, rule)
    train_rules_ind.append(train_rule_ind)
    valid_rules_ind.append(valid_rule_ind)
    test_rules_ind.append(test_rule_ind)

    return train_rules_ind, valid_rules_ind, test_rules_ind


def extract_ind(input1, input2, input3, rule):
    print ("loading data")
    attribute_top = np.loadtxt(open('./ind/ExFashion/top_'+rule[0]+'.txt', "r"), dtype="str", delimiter="\t")
    attribute_bottom = np.loadtxt(open('./ind/ExFashion/bottom_'+rule[0]+'.txt', "r"), dtype="str", delimiter="\t")
    ind = []
    for i in range(input1.shape[0]):
        i_id = str(int(input1[i]))
        j_id = str(int(input2[i]))
        k_id = str(int(input3[i]))
        i_index = np.where(attribute_top == i_id)[0][0]
        j_index = np.where(attribute_bottom == j_id)[0][0]
        k_index = np.where(attribute_bottom == k_id)[0][0]
        i_attribute = attribute_top[i_index][1]
        j_attribute = attribute_bottom[j_index][1]
        k_attribute = attribute_bottom[k_index][1]

        # whether 3 items have material lable
        if j_attribute != 0 and k_attribute != 0:
            # get frequence
            freq_mask = frequence(i_attribute, j_attribute, k_attribute, rule)
            ind.append(freq_mask)
        else:
            ind.append([0,0,0])
    ind = np.array(ind)
    count1 = 0
    count2 = 0
    for i in range(ind.shape[0]):
        if ind[i][0] != 0:
            if ind[i][1] > 0.5:
                count1 += 1
            if ind[i][2] > 0.5:
                count2 += 1
    print ind.shape
    print 'mun110: %i' % count1
    print 'mun101: %i' % count2
    return ind

def frequence(i_m, j_m, k_m, rule):
    list = np.array(np.loadtxt(open('./ind/ExFashion/'+rule[0]+'_frequence.txt', "r"), delimiter="\t", dtype=str))
    list_t = list[:int(rule[1]), 0]
    list_b = list[:int(rule[1]), 1]
    freq = list[:int(rule[1]), 2]
    list_t_e = list[int(rule[2]):, 0]
    list_b_e = list[int(rule[2]):, 1]
    freq_e = list[int(rule[2]):, 2]
    ind = []
    mask = [1]
    # find frequence of ij
    for i in range(len(list_t)):
        if i_m == list_t[i]:
            if j_m == list_b[i]:
                ind.append(int(freq[i]))
                flag = 1
    for i in range(len(list_t_e)):
        if i_m == list_t_e[i]:
            if j_m == list_b_e[i]:
                ind.append(int(freq_e[i]))

    # find frequence of ik
    for i in range(len(list_t)):
        if i_m == list_t[i]:
            if k_m == list_b[i]:
                ind.append(int(freq[i]))
    for i in range(len(list_t_e)):
        if i_m == list_t_e[i]:
            if k_m == list_b_e[i]:
                ind.append(int(freq_e[i]))

    if len(ind) == 2:
        ind = norm(ind)
        if ind[0] == 0.5:
            mask = [0, 0, 0]
        else:
            mask.append(ind[0])
            mask.append(ind[1])
    else:
        mask = [0, 0, 0]

    return mask

