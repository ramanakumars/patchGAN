from torch.nn.parameter import Parameter


class Transferable():
    def __init__(self):
        super(Transferable, self).__init__()

    def load_transfer_data(self, state_dict):
        own_state = self.state_dict()
        count = 0
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if param.shape == own_state[name].data.shape:
                own_state[name].copy_(param)
                count += 1

        if count > 0:
            print(f"Loaded {count} weights out of {len(state_dict)}")
        else:
            raise InvalidCheckpointError("Could not load transfer weights")


class InvalidCheckpointError(Exception):
    pass
