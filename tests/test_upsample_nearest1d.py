import random
import torch
import flag_gems
import torch.nn.functional as F

    
# testing
while True:
    L_in = random.randint(100, 1000)
    L_out = random.randint(100, 1000)
    scale_factor = float(L_out) / float(L_in)
    gems_input = torch.randn((2, 3, L_in), device=flag_gems.device)
    torch_input = gems_input.clone()
    with flag_gems.use_gems():
        gems_output = flag_gems.upsample_nearest1d(
            gems_input,
            output_size=(L_out,),
        )
    torch_output = F.interpolate(
        torch_input,
        size=(L_out,),
        mode="nearest",
    )
    if not torch.allclose(gems_output, torch_output, atol=1e-5, rtol=1e-4):
        print(f"Test failed for L_in={L_in}, L_out={L_out}")
        print("GEMS output:", gems_output)
        print("Torch output:", torch_output)
        break
    else:
        print(f"Test passed for L_in={L_in}, L_out={L_out}")
        
    with flag_gems.use_gems():
        gems_output = flag_gems.upsample_nearest1d(
            gems_input,
            scales=scale_factor,
        )
    torch_output = F.interpolate(
        torch_input,
        scale_factor=scale_factor,
        mode="nearest",
    )
    if not torch.allclose(gems_output, torch_output, atol=1e-5, rtol=1e-4):
        print(f"Test failed for L_in={L_in}, L_out={L_out} with scale_factor")
        print("GEMS output:", gems_output)
        print("Torch output:", torch_output)
        break
    else:
        print(f"Test passed for L_in={L_in}, L_out={L_out} with scale_factor")
    
    