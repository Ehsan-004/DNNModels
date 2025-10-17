import torch
from UNet import DoubleConv, Down, Up, OutConv, UNet


def test_double_conv():    
    model = DoubleConv(3, 64)
    data = torch.randn(1, 3, 224, 224)
    out = model(data)
    assert out.shape[1] == 64


def test_down():
    model = Down(3, 64)
    
    data1 = torch.randn(1, 3, 224, 224)
    data2 = torch.randn(1, 3, 225, 225)
    
    out1 = model(data1)
    out2 = model(data2)
    
    assert out1.shape[1] == 64
    assert out1.shape[2] == 112
    assert out1.shape[3] == 112
    assert out2.shape[2] == 112
    assert out2.shape[3] == 112


def test_up():
    model1 = Up(64, 32, 'upconv')
    model2 = Up(64, 32, 'interp')
    
    conc1 = torch.randn(1, 32, 200, 200)
    # conc2 = torch.randn(1, 64, 101, 101)
    
    data11 = torch.randn(1, 64, 100, 100)
    # data12 = torch.randn(1, 64, 101, 101)
    data21 = torch.randn(1, 64, 100, 100)
    # data22 = torch.randn(1, 64, 101, 101)
    
    out11 = model1(data11, conc1)
    # out12 = model1(data12, conc2)
    out21 = model2(data21, conc1)
    # out22 = model2(data22, conc2)
    
    assert out11.shape[1] == 32
    assert out11.shape[2] == 200
    assert out11.shape[3] == 200
    
    # print("\nshape out12")
    # print(out12.shape)
    
    assert out21.shape[1] == 32
    assert out21.shape[2] == 200
    assert out21.shape[3] == 200
    
    # print("shape out22")
    # print(out22.shape)
    

def test_out_conv():
    model = OutConv(3, 64)
    data = torch.randn(1, 3, 224, 224)
    out = model(data)
    
    assert out.shape[1] == 64
    assert out.shape[2] == 224
    assert out.shape[3] == 224


def test_unet():
    model = UNet(3, 23, 4, 64)
    data1 = torch.randn(1, 3, 224, 224)
    data2 = torch.randn(1, 3, 225, 225)
    out1 = model(data1)
    out2 = model(data2)
    
    assert out1.shape[1] == 23
    assert out1.shape[2] == 224
    assert out1.shape[3] == 224
    
    assert out2.shape[1] == 23
    assert out2.shape[2] == 225
    assert out2.shape[3] == 225

    
# def unet_test():
if __name__ == "__main__":
    print("[test] double conv", end="")
    test_double_conv()
    print(" | done")
    
    print("[test] down", end="")
    test_down()
    print(" | done")
    
    print("[test] up", end="")
    test_up()
    print(" | done")
    
    print("[test] out conv", end="")
    test_out_conv()
    print(" | done")
    
    print("[test] unet", end="")
    test_unet()
    print(" | done")
