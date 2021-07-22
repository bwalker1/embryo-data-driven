load('scData/skin_data.mat');

genes_used = importdata('../input_data/SpatialRef/genes_in_paper.txt');

num_cell_types = max(cell_id);

mean_expr = [];
c = 1;
processed_genes = {};

for i = 1:length(genes_used)
    gene = genes_used{i};
    %disp(gene);
    for j = 1:length(allgenes)
        if strcmp(gene,allgenes{j})
            counts_for_gene = count_matrix(j, :);
            % get mean expression by cell type
            for k = 1:num_cell_types
                expr = mean(counts_for_gene(cell_id == k));
                mean_expr(k, c) = expr;
            end
            % normalize expression over row
            mean_expr(:, c) = mean_expr(:, c)/max(mean_expr(:, c));
            processed_genes{c} = gene;
            c = c+1;
        end
    end
end

writecell(processed_genes,'processed_genes.csv');
writematrix(mean_expr,'mean_expr.csv');
