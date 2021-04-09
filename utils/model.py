from models.codegnndualfcinfo import CodeGNNDualFcInfo
from models.codegnndualfcinfo_fc import CodeGNNDualFcInfo_Fc
from models.codegnndualfcinfo_gnn import CodeGNNDualFcInfo_Gnn
from models.codegnndualfcinfo_info import CodeGNNDualFcInfo_Info

def create_model(modeltype, config,hightoken=None):
    mdl = None
    if hightoken == None:
        if modeltype == 'codegnndualfcinfo':
            mdl = CodeGNNDualFcInfo(config)
        elif modeltype == 'codegnndualfcinfo_fc':
            mdl = CodeGNNDualFcInfo_Fc(config)
        elif modeltype == 'codegnndualfcinfo_gnn':
            mdl = CodeGNNDualFcInfo_Gnn(config)
        elif modeltype == 'codegnndualfcinfo_info':
            mdl = CodeGNNDualFcInfo_Info(config)
        else:
            print("{} is not a valid model type".format(modeltype))
            exit(1)
    else:
        if modeltype == 'codegnndualfcinfo':
            mdl = CodeGNNDualFcInfo(config,hightoken)
        elif modeltype == 'codegnndualfcinfo_fc':
            mdl = CodeGNNDualFcInfo_Fc(config)
        elif modeltype == 'codegnndualfcinfo_gnn':
            mdl = CodeGNNDualFcInfo_Gnn(config)
        elif modeltype == 'codegnndualfcinfo_info':
            mdl = CodeGNNDualFcInfo_Info(config)
        else:
            print("{} is not a valid model type".format(modeltype))
            exit(1)

    return mdl.create_model()
