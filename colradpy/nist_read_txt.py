import numpy as np


def _text(value):
    """String of a genfromtxt entry, which is bytes on NumPy < 2 and str on NumPy >= 2."""
    if(isinstance(value, bytes)):
        return value.decode('UTF-8')
    return str(value)


def get_nist_txt(path,el,ch):

    tmp = np.genfromtxt(path+el.lower()+'_'+str(ch),dtype=None,delimiter=',')
    levels = []
    for i in range(0,len(tmp)):
        tmp_dic = {}
        tmp_dic['conf']   = _text(tmp[i][0])
        tmp_dic['term']   = _text(tmp[i][1])
        tmp_dic['j_val']  = _text(tmp[i][2])
        tmp_dic['energy'] = tmp[i][3]
        levels.append(tmp_dic)

    return levels
