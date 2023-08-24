var input = [...]

input = input.map(v => v.trim())

var output = [...]
output = output.map(v => v.trim())

copy(input.map((v, i) => ({input: v, output: output[i]})))