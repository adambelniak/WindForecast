import torch
import torch.nn as nn
from einops import repeat, rearrange

from .extra_layers import ConvBlock, Flatten

class Time2Vec(nn.Module):
    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            x = torch.diag_embed(x)
            # x.shape = (bs, sequence_length, input_dim, input_dim)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # x_affine.shape = (bs, sequence_length, input_dim, time_embed_dim)
            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1
            )
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
            # x_output.shape = (bs, sequence_length, input_dim * time_embed_dim)
        else:
            x_output = x
        return x_output

class Embedding(nn.Module):
    def __init__(
        self,
        d_input,
        d_time_features,
        d_model,
        time_emb_dim=6,
        method="spatio-temporal",
        downsample_convs=0,
        start_token_len=0,
        null_value=None,
        pad_value=None,
        is_encoder: bool = True,
        position_emb="abs",
        data_dropout=None,
        max_seq_len=None,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
    ):
        super().__init__()

        assert method in ["spatio-temporal", "temporal"]
        if data_dropout is None:
            self.data_drop = lambda y: y
        else:
            self.data_drop = data_dropout

        self.method = method


        time_dim = time_emb_dim * d_time_features
        self.time_emb = Time2Vec(d_time_features, embed_dim=time_dim)

        assert position_emb in ["t2v", "abs"]
        self.max_seq_len = max_seq_len
        self.position_emb = position_emb
        if self.position_emb == "t2v":
            # standard periodic pos emb but w/ learnable coeffs
            self.local_emb = Time2Vec(1, embed_dim=d_model + 1)
        elif self.position_emb == "abs":
            # lookup-based learnable pos emb
            assert max_seq_len is not None
            self.local_emb = nn.Embedding(
                num_embeddings=max_seq_len, embedding_dim=d_model
            )

        y_emb_inp_dim = d_input if self.method == "temporal" else 1
        self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim, d_model)

        if self.method == "spatio-temporal":
            self.space_emb = nn.Embedding(num_embeddings=d_input, embedding_dim=d_model)
            split_length_into = d_input
        else:
            split_length_into = 1

        self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [ConvBlock(split_length_into, d_model) for _ in range(downsample_convs)]
        )

        self.d_model = d_model
        self.null_value = null_value
        self.pad_value = pad_value
        self.is_encoder = is_encoder

        # turning off parts of the embedding is only really here for ablation studies
        self.use_val = use_val
        self.use_time = use_time
        self.use_given = use_given
        self.use_space = use_space

    def __call__(self, input: torch.Tensor, dates: torch.Tensor):
        if self.method == "spatio-temporal":
            emb = self.spatio_temporal_embed
        else:
            emb = self.temporal_embed
        return emb(input=input, dates=dates)

    def make_mask(self, y):
        # we make padding-based masks here due to outdated
        # feature where the embedding randomly drops tokens by setting
        # them to the pad value as a form of regularization
        if self.pad_value is None:
            return None
        return (y == self.pad_value).any(-1, keepdim=True)

    def temporal_embed(self, input: torch.Tensor, dates: torch.Tensor):
        bs, length, d_input = input.shape

        # protect against true NaNs. without
        # `spatio_temporal_embed`'s multivariate "Given"
        # concept there isn't much else we can do here.
        # NaNs should probably be set to a magic number value
        # in the dataset and passed to the null_value arg.
        input = torch.nan_to_num(input)

        if self.is_encoder:
            # optionally mask the context sequence for reconstruction
            input = self.data_drop(input)
        mask = self.make_mask(input)

        # position embedding ("local_emb")
        local_pos = torch.arange(length).to(input.device)
        if self.position_emb == "t2v":
            # first idx of Time2Vec output is unbounded so we drop it to
            # reuse code as a learnable pos embb
            local_emb = self.local_emb(
                local_pos.view(1, -1, 1).repeat(bs, 1, 1).float()
            )[:, :, 1:]
        elif self.position_emb == "abs":
            assert length <= self.max_seq_len
            local_emb = self.local_emb(local_pos.long().view(1, -1).repeat(bs, 1))

        # time embedding (Time2Vec)
        if not self.use_time:
            dates = torch.zeros_like(dates)
        time_emb = self.time_emb(dates)
        if not self.use_val:
            input = torch.zeros_like(input)
        # concat time emb to value --> FF --> val_time_emb
        val_time_inp = torch.cat((time_emb, input), dim=-1)
        val_time_emb = self.val_time_emb(val_time_inp)

        # "given" embedding. not important for temporal emb
        # when not using a start token
        given = torch.ones((bs, length)).long().to(input.device)
        if not self.is_encoder and self.use_given:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given)

        emb = local_emb + val_time_emb + given_emb

        if self.is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        # space emb not used for temporal method
        space_emb = torch.zeros_like(emb)
        var_idxs = None
        return emb, space_emb, var_idxs, mask

    def spatio_temporal_embed(self, input: torch.Tensor, dates: torch.Tensor):
        # full spatiotemopral emb method. lots of shape rearrange code
        # here to create artificially long (length x dim) spatiotemporal sequence
        batch, length, d_input = input.shape

        # position emb ("local_emb")
        local_pos = repeat(
            torch.arange(length).to(input.device), f"length -> {batch} ({d_input} length)"
        )
        if self.position_emb == "t2v":
            # periodic pos emb
            local_emb = self.local_emb(local_pos.float().unsqueeze(-1).float())[
                :, :, 1:
            ]
        elif self.position_emb == "abs":
            # lookup pos emb
            local_emb = self.local_emb(local_pos.long())

        # time emb
        if not self.use_time:
            dates = torch.zeros_like(dates)
        dates = torch.nan_to_num(dates)
        dates = repeat(dates, f"batch len x_dim -> batch ({d_input} len) x_dim")
        time_emb = self.time_emb(dates)

        # protect against NaNs in y, but keep track for Given emb
        true_null = torch.isnan(input)
        input = torch.nan_to_num(input)
        if not self.use_val:
            input = torch.zeros_like(input)

        # keep track of pre-dropout y for given emb
        input_original = input.clone()
        input_original = Flatten(input_original)
        input = self.data_drop(input)
        input = Flatten(input)
        mask = self.make_mask(input)

        # concat time_emb, y --> FF --> val_time_emb
        val_time_inp = torch.cat((time_emb, input), dim=-1)
        val_time_emb = self.val_time_emb(val_time_inp)

        # "given" embedding
        if self.use_given:
            given = torch.ones((batch, length, d_input)).long().to(input.device)  # start as True
            if not self.is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0  # (False)

            # if y was NaN, set Given = False
            given *= ~true_null

            # flatten now to make the rest easier to figure out
            given = rearrange(given, "batch len dy -> batch (dy len)")

            # use given embeddings to identify data that was dropped out
            given *= (input == input_original).squeeze(-1)

            if self.null_value is not None:
                # mask null values that were set to a magic number in the dataset itself
                null_mask = (input != self.null_value).squeeze(-1)
                given *= null_mask

            given_emb = self.given_emb(given)
        else:
            given_emb = 0.0

        val_time_emb = local_emb + val_time_emb + given_emb

        if self.is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # space embedding
        var_idx = repeat(
            torch.arange(d_input).long().to(input.device), f"dy -> {batch} (dy {length})"
        )
        var_idx_true = var_idx.clone()
        if not self.use_space:
            var_idx = torch.zeros_like(var_idx)
        space_emb = self.space_emb(var_idx)

        return val_time_emb, space_emb, var_idx_true, mask
