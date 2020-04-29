class MyConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, init_mode='xavier', std_dev=0.05):
        
        super(MyConv2d, self).__init__()
       
        self.weight = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        if init_mode == "gaussian":
          normal_(self.weight, std=std_dev)
        else:
          xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        self.bias = Parameter(torch.empty(out_channels))
        nn.init.normal_(self.bias, std=std_dev)

    def forward(self, X):
        # return F.conv2d(X, weight=self.weight, bias=self.bias)
        return MyConv2d.myconv2d(X, self.weight, self.bias)

    @staticmethod
    def myconv2d(X, weight, bias):
        '''
        X.shape is [batch_size, in_channels, image_height, image_width]
        weight.shape is [out_channels, in_channels, kernel_height, kernel_width]
        bias.shape is [out_channels]
        return shape [batch_size, out_channels, out_height, out_width]
        '''
        image_height, image_width = tuple(X.shape[-2:])
        out_height, out_width = image_height-2, image_width-2
        in_channels, out_channels = X.shape[1], weight.shape[0]
        
        input_tensor_unfolded = F.unfold(X, 3).permute(0,2,1)
        weight_tensor_unfolded = weight.view(-1,in_channels*9).transpose(0,1)
        res = torch.matmul(input_tensor_unfolded, weight_tensor_unfolded) + bias
        return res.permute(0,2,1).view(-1,out_channels,out_height, out_width)
