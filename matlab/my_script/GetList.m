function list = GetList(fn)
  fid2 = fopen(fn, 'r');
  list = textscan(fid2, '%s');
  list = list{1};
  fclose(fid2);
end