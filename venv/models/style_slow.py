import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_features).detach()
        self.loss = 0

    def gram_matrix(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class StyleTransfer(nn.Module):
    def __init__(self, model):
        super(StyleTransfer, self).__init__()
        self.base_model = model


    def make_model(self, content_img, style_img):
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        content_losses = []
        style_losses = []
        style_model = nn.Sequential()

        for layer in self.base_model.named_children():
            style_model.add_module(layer[0], layer[1])
            if layer[0] in content_layers:
                target = style_model(content_img).detach()
                content_loss = ContentLoss(target)
                style_model.add_module('content_loss_{}'.format(layer[0][-1]), content_loss)
                content_losses.append(content_loss)
            if layer[0] in style_layers:
                target_feature = style_model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                style_model.add_module('style_loss_{}'.format(layer[0][-1]), style_loss)
                style_losses.append(style_loss)
        return style_model, content_losses, style_losses

    def get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def transform(self, content_img, style_img, num_steps=300, style_weight=100000, content_weight=1):
        style_model, content_losses, style_losses = self.make_model(content_img, style_img)

        input_img = content_img.clone()

        input_img.requires_grad_(True)
        style_model.eval()
        style_model.requires_grad_(False)
        optimizer = self.get_input_optimizer(input_img)

        run = [0]
        while run[0] <= num_steps:
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                optimizer.zero_grad()
                style_model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img


vgg_small_model = torch.jit.load('C:/Users/nkuzn/PycharmProjects/StyleBot2/venv/models/model_scripted.pt')
style_model = StyleTransfer(vgg_small_model)