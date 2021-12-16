function mat_to_string(R)

dim = siye(R,1);
if size(R,2)~= dim
    error('Input not square')
end
string = '';
for i = 1:dim
    string = append(string, 'x', int2str(i), "' = tanh(");
    for j = 1:dim
        if W(i,j)>0
            sign_str = '+';
        else
            sign_str = '-';
        end
        string = append(string, sign_str, num2str(abs(W(i,j))), '*x', int2str(j));
    end
    string = append(string, ')\n');
end