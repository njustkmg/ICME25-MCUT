import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import copy
import math
Parameter.fast = None

class CMML_normal_Fusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(CMML, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, y):
        audio_attention = self.softmax(x.clone())
        visual_attention = self.softmax(y.clone())
        sum_modal =  audio_attention+visual_attention
        audio_attention = audio_attention/sum_modal
        visual_attention = visual_attention/sum_modal
        supervise_feature_hidden = audio_attention * x + visual_attention * y
        output = self.fc_out(supervise_feature_hidden)
        return x, y, output
    
class Sum_normal_Fusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output

class Concat_normal_Fusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(Concat_normal_Fusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        feature = torch.cat((x, y), dim=1)
        output = self.fc_out(feature)
        return output

class LSTM_normal_Fusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(LSTMFusion, self).__init__()
        self.lstm = nn.LSTM(input_dim, 512, 1, batch_first=True)
        self.fc_out = nn.Linear(512, output_dim)  # Assuming concatenated features from BERT and ResNet

    def forward(self, x, y):
        fused_features = torch.cat((x,y), dim=1)
        out, _ = self.lstm(fused_features.unsqueeze(1))
        lstm_output = out[:, -1, :]
        output = self.fc_out(lstm_output)
        return x,y,output

class FiLM_normal_Fusion(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class CMML(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(CMML, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, y,mode=0):
        if mode==0:
            audio_attention = self.softmax(x.clone())
            visual_attention = self.softmax(y.clone())
            sum_modal =  audio_attention+visual_attention
            audio_attention = audio_attention/sum_modal
            visual_attention = visual_attention/sum_modal
            supervise_feature_hidden = audio_attention * x + visual_attention * y
            output = self.fc_out(supervise_feature_hidden)
            return output
        else:
            x_out = self.fc_out(x)
            y_out = self.fc_out(y)
            return x_out, y_out
        
class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, mode =0):
        x_out = self.fc_x(x) 
        y_out = self.fc_y(y)
        output = (x_out+ y_out)/2
        if mode==0:
            return output
        else:
            return x_out, y_out
        
class LSTMFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(LSTMFusion, self).__init__()
        self.lstm = nn.LSTM(input_dim, 512, 1, batch_first=True)
        self.fc_out = nn.Linear(512, output_dim)  # Assuming concatenated features from BERT and ResNet

    def forward(self, x, y, mode):
        if mode==0:
            fused_features = torch.cat((x,y), dim=1)
            out, _ = self.lstm(fused_features.unsqueeze(1))
            lstm_output = out[:, -1, :]
            output = self.fc_out(lstm_output)
            return output
        else:
            x_padding = torch.zeros_like(x, device=x.device)
            y_padding = torch.zeros_like(y, device=y.device)
            feature_x = torch.cat((x, y_padding), dim=1)
            feature_y = torch.cat((x_padding, y), dim=1)
            x_out, _ = self.lstm(feature_x.unsqueeze(1))
            y_out, _ = self.lstm(feature_y.unsqueeze(1))
            x_out = x_out[:, -1, :]
            y_out = y_out[:, -1, :]
            x_out = self.fc_out(x_out)
            y_out = self.fc_out(y_out)
            return x_out, y_out
    
class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, mode =0 ):
        x_padding = torch.zeros_like(x, device=x.device)
        y_padding = torch.zeros_like(y, device=y.device)
        
        feature = torch.cat((x, y), dim=1)
        feature_x = torch.cat((x, y_padding), dim=1)
        feature_y = torch.cat((x_padding, y), dim=1)
        if mode==0:
            output = self.fc_out(feature)
            return output
        else:
            if self.fc_out.weight.fast is not None:
                x_out = torch.mm(feature_x, torch.transpose(self.fc_out.weight.fast, 0, 1))+ self.fc_out.bias 
                y_out = torch.mm(feature_y, torch.transpose(self.fc_out.weight.fast, 0, 1))+ self.fc_out.bias 
            else:
                x_out = self.fc_out(feature_x)
                y_out = self.fc_out(feature_y)
            return x_out, y_out
      
class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y, mode=0):
        if mode==0:
            if self.x_film:
                film = x
                to_be_film = y
            else:
                film = y
                to_be_film = x

            gamma, beta = torch.split(self.fc(film), self.dim, 1)

            output = gamma * to_be_film + beta
            output = self.fc_out(output)
            return output
        else:
            x_out = self.fc_out(x)
            y_out = self.fc_out(y)
            return x_out, y_out