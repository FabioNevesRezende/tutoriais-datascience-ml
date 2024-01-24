# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import charset_normalizer

# set seed for reproducibility
np.random.seed(0)


# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
print(type(before))

after = before.encode("utf-8", errors="replace")

print(type(after))

print(before)

print(after)

print(after.decode("utf-8"))

# try to decode our bytes with the ascii encoding
# print(after.decode("ascii"))



# start with a string
before2 = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after2 = before2.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after2)
print(after2.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(

# error
# kickstarter_2016 = pd.read_csv("../datasets/ks-projects-201612.csv")

with open("../datasets/ks-projects-201612.csv",'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(100000))

print(result)

kickstarter_2016 = pd.read_csv("../datasets/ks-projects-201612.csv", encoding="Windows-1252")

print(kickstarter_2016.head())

kickstarter_2016.to_csv("../datasets/ks-projects-201801-utf8.csv")