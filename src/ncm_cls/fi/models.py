import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")]
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):
    channel_config = {}
    for ss in s.split(","):
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


class Encoder(nn.Module):
    def __init__(self, block_config_str, channel_config_str, hidden_size):
        super().__init__()

        self.in_conv = nn.Conv2d(3, 32, 3, stride=1, padding=3, bias=False)

        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, down_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue
            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=down_rate,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)

        # Latents
        self.mu = nn.Conv2d(channel_config[1],  hidden_size, 1, bias=False)
        self.logvar = nn.Conv2d(channel_config[1],  hidden_size, 1, bias=False)

    def forward(self, input):
        x = self.in_conv(input)
        x = self.block_mod(x)
        return self.mu(x), self.logvar(x)

    def get_rep(self, input):
        x = self.in_conv(input)
        x = self.block_mod(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_res, block_config_str, channel_config_str, c_dim, l_dim):
        super().__init__()
        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []

        self.i_size = c_dim + l_dim

        blocks.append(
            nn.Conv2d(self.i_size, channel_config[1], 1, bias=False)
        )

        for _, (res, up_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue

            if up_rate is not None:
                blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=None,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)
        self.last_conv = nn.Conv2d(channel_config[input_res], 3, 5, stride=1)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight,
                                     gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, c, u):
        c = 2 * c - 1
        x = torch.cat((c, u), dim=1)
        x = self.block_mod(x)
        x = self.last_conv(x)
        # return torch.sigmoid(x)
        return torch.tanh(x)


"""
models for unets in ddpm
"""


class ResidualConvBlock(nn.Module):
    """Convolutional Residual Block"""

    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualConvBlock, self).__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.handle_channel = None
        if self.is_res and not self.same_channels:
            self.handle_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x_out = self.conv2(self.conv1(x))
        if not self.is_res:
            return x_out
        else:
            if self.handle_channel:
                x_out = x_out + self.handle_channel(x)
            else:
                x_out = x_out + x
            return x_out / 1.414  # normalize


class UNetDown(nn.Module):
    """UNet downward block (i.e., contracting block)"""

    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels, True),  # use skip-connection
                  ResidualConvBlock(out_channels, out_channels),
                  nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """UNet upward block (i.e., expanding block)"""

    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),  # double h & w sizes
            ResidualConvBlock(out_channels, out_channels, True),  # use residual-connection
            ResidualConvBlock(out_channels, out_channels)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_skip):
        return self.model(torch.cat([x, x_skip], dim=1))


class EmbedFC(nn.Module):
    """Fully Connected Embedding Layer"""

    def __init__(self, input_dim, hidden_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)[:, :, None, None]


class ContextUnet(nn.Module):
    """Context UNet model
    Args:
        in_channels (int): Number of channels in the input image
        height (int): Height of the input image
        width (int): Width of the input image
        n_feat (int): Number of initial features i.e., hidden channels to which
            the input image be transformed
        n_cfeat (int): Number of context features i.e., class categories
        n_downs (int): Number of down (and up) blocks of UNet. Default: 2
    """

    def __init__(self, in_channels, height, width, n_feat, n_cfeat, n_downs=2):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.n_downs = n_downs

        # Define initial convolution
        self.init_conv = ResidualConvBlock(in_channels, n_feat, True)

        # Define downward unet blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(2 ** i * n_feat, 2 ** (i + 1) * n_feat))

        # Define at the center layers
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((height // 2 ** len(self.down_blocks), width // 2 ** len(self.down_blocks))),
            nn.GELU())
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                2 ** n_downs * n_feat,
                2 ** n_downs * n_feat,
                (height // 2 ** len(self.down_blocks), width // 2 ** len(self.down_blocks))),
            nn.GroupNorm(8, 2 ** n_downs * n_feat),
            nn.GELU()
        )

        # Define upward unet blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downs, 0, -1):
            self.up_blocks.append(UNetUp(2 ** (i + 1) * n_feat, 2 ** (i - 1) * n_feat))

        # Define final convolutional layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, in_channels, 1, 1)
        )

        # Define time & context embedding blocks
        self.timeembs = nn.ModuleList([EmbedFC(1, 2 ** i * n_feat) for i in range(n_downs, 0, -1)])
        self.contextembs = nn.ModuleList([EmbedFC(n_cfeat, 2 ** i * n_feat) for i in range(n_downs, 0, -1)])

    def forward(self, x, t, c):
        x = self.init_conv(x)
        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0:
                downs.append(down_block(x))
            else:
                downs.append(down_block(downs[-1]))
        up = self.up0(self.to_vec(downs[-1]))
        for up_block, down, contextemb, timeemb in zip(self.up_blocks, downs[::-1], self.contextembs, self.timeembs):
            up = up_block(up * contextemb(c) + timeemb(t), down)
        return self.final_conv(torch.cat([up, x], axis=1))
