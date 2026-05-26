classdef item
    properties
        weight 
        value 
    end

    methods
        function obj = item(weight, value)
            if nargin >= 1
                obj.weight = weight;
            end
            if nargin >= 2
                obj.value = value;
            end
        end    
    end
end
