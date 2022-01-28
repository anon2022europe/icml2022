from ggg.data.dense.PEAWGANDenseStructureData import PEAWGANDenseStructureData

ds = PEAWGANDenseStructureData(dataset="CommunitySmall_20", zero_pad=True)

for i in range(3):
    x, A = ds[i]
    print(A, ds.max_N)
