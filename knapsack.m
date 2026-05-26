classdef knapsack
    properties
        max_weight
        load_weight
        load_value
    end

    methods
        function obj = knapsack(set_max_weight, start_load_weight, start_load_value)
            if nargin >= 1
                obj.max_weight = set_max_weight;
            else
                obj.max_weight = 0;
            end
            if nargin >= 2
                obj.load_weight = start_load_weight;
            else
                obj.load_weight = 0;
            end
            if nargin >= 3
                obj.load_value = start_load_value;
            else
                obj.load_value = 0;
            end
        end
    end
end



