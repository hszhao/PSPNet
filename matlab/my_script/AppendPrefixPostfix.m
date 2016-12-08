function out = AppendPrefixPostfix(in, prefix, postfix)
    out = cell(size(in));
    
    for i = 1 : numel(in)
        out{i} = [prefix, in{i}, postfix];
    end
end