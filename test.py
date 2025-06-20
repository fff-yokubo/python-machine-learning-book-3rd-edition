
nsize = 5



origins = [(0,0)]
for i in range(nsize-2):
  origins += [(i+1,0),(0,i+1)]

print(origins)

origins = [(0,nsize-1)]
for i in range(nsize-2):
  origins += [(i+1,nsize-1),(0,nsize-i-2)]
#   # origins += [(i+1,nsize-1)]
#   origins += [(0,nsize-1-i-1)]

print(origins)