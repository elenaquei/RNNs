function [string_system] = mat2CAPDstr(W)

n_variables = size(W,1);
string_system = 'par: gamma; time: t; var: ';
for i = 1:n_variables
    xi = string_variable(i, n_variables);
    if i > 1
        string_system = append(string_system, ', ', xi);
    end
end
string_system = append(string_system,'; fun:');
for i = 1:n_variables
    string_system = append(string_system, ' tanh(');
    for j = 1:n_variables
        xj = string_variable(j, n_variables);
        new_term = '';
        if i == j 
            new_term = append(new_term, '-gamma * ', xj);
        end
        if W(i,j)>=0
            sign_Wij = '+';
        else
            sign_Wij = '-';
        end
            
        new_term = append(new_term, sign_Wij,sprintf('%.3f * %s ', abs(W(i,j)), xj));
        string_system = append(string_system, new_term);
    end
    if i < n_variables
        string_system = append(string_system, '),');
    else
        string_system = append(string_system, ');');
    end
end


end

function xj = string_variable(j, n_variables)

xj = num2str(j);
required_length = floor(log10(n_variables));
while length(xj)<required_length
    xj = append('0',xj);
end
xj = append('x', xj);

end