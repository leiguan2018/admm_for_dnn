import time

def get_absolute_path():
    path = "/home/lei/project/admm_dnn/admm_dnn_code/"
    return path


def save_results_to_disk(name, runtime_list, train_acc_list, val_acc_list):
    ab_path = get_absolute_path()
    filename = ab_path + 'final_result/' + name + '_result_' + time.strftime("%Y%m%d", time.localtime())
    file = open(filename, 'w')
    file.write(str(runtime_list))
    file.write('\r\n')
    file.write(str(train_acc_list))
    file.write('\r\n')
    file.write(str(val_acc_list))
    file.write('\r\n')
    file.close()
