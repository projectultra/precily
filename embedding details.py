# Replace 'embedding' with the name of your file
filename = 'embedding'
import os
# Get the size of the file in bytes
size = os.path.getsize(filename)
print(f"File size: {size} bytes")

# Get the creation time of the file
creation_time = os.path.getctime(filename)
print(f"Creation time: {creation_time}")

# Get the modification time of the file
modification_time = os.path.getmtime(filename)
print(f"Modification time: {modification_time}")