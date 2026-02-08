% 指定 .mat 文件路径
matFilePath = 'G:\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\mushroom\预处理完成\mushroom.mat';

% 加载 .mat 文件数据
data = load(matFilePath);

% 获取 .mat 文件中变量的字段名
fieldNames = fieldnames(data);

% 选择要写入的数据 (假设数据存储在第一个字段中)
selectedData = data.(fieldNames{1});

% 指定输出 .xlsx 文件路径
xlsxFilePath = 'G:\148个整理好的UCI数据集及相应代码\148个整理好的UCI数据集及相应代码\mushroom\预处理完成\mushroom.xlsx';

% 检查数据格式并写入 .xlsx 文件
if isnumeric(selectedData)
    % 如果数据是数值矩阵
    writematrix(selectedData, xlsxFilePath);
elseif iscell(selectedData)
    % 如果数据是元胞数组
    writecell(selectedData, xlsxFilePath);
elseif istable(selectedData)
    % 如果数据是表格
    writetable(selectedData, xlsxFilePath);
else
    error('数据格式不支持直接写入 Excel 文件');
end

disp('数据已成功写入 .xlsx 文件');
