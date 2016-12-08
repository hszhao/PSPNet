function out = LoadBinFile(fn, type)
% load binary file

fid = fopen(fn, 'rb');

row = fread(fid, 1, 'int32');
col = fread(fid, 1, 'int32');
channel = fread(fid, 1, 'int32');
num = fread(fid, 1, 'int32');

num_ele = row*col*channel*num;

if strcmp(type, 'int32')
    out = fread(fid, num_ele, 'int32');
elseif strcmp(type, 'single')
    out = fread(fid, num_ele, 'single');   
elseif strcmp(type, 'uint8')
    out = fread(fid, num_ele, 'uint8');
else
    error('wrong type')
end

out = reshape(out, [row, col, channel, num]);

fclose(fid);