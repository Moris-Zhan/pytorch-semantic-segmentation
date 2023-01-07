import io
from contextlib import redirect_stdout
from torchinfo import summary
import os

def write_info(output, model, shape, fn): 
    f = io.StringIO()
    with redirect_stdout(f):        
        summary(model, input_size=shape )
    lines = f.getvalue()
    print("".join(lines))

    with open( os.path.join(output, fn) ,"w") as f:
        [f.write(line) for line in lines]