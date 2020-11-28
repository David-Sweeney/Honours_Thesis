import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

e_field = pd.read_csv('/import/tintagel3/snert/david/Reverse_Injection/run1/bptmp.fld', sep='\s+', header=None, skiprows=4)
print(e_field.shape)
plt.imshow(e_field)
plt.colorbar()
plt.title('Electric Field')
plt.savefig('/import/tintagel3/snert/david/Reverse_Injection/E_Field.png')
plt.show()
