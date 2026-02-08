clear; clc;

% -------------------- CHOOSE DATASET (one at a time) --------------------
% Example:
%   DATASET = 'pass_fail';
% Then you need a file: /datasets/run_pass_fail.m
DATASET = 'sonar'; % <-- you will change this each time

% -------------------- GLOBAL CONFIG --------------------
CFG = struct();
CFG.N = 30;              % population size
CFG.MaxIt = 30;          % iterations
CFG.runs = 10;           % repeated runs for statistics
CFG.test_ratio = 0.50;   % holdout test split
CFG.seed0 = 42;          % base random seed

% -------------------- ADD PATHS --------------------
RUN_DIR  = fileparts(mfilename('fullpath'));
ROOT_DIR = fileparts(RUN_DIR);

addpath(genpath(fullfile(ROOT_DIR, 'klsr')));
addpath(genpath(fullfile(ROOT_DIR, 'fs')));

% run subfolders (we will create scripts here)
addpath(genpath(fullfile(RUN_DIR, 'core')));
addpath(genpath(fullfile(RUN_DIR, 'clean')));
addpath(genpath(fullfile(RUN_DIR, 'datasets')));

% results folder (auto create)
RESULTS_DIR = fullfile(RUN_DIR, 'results');
if ~exist(RESULTS_DIR, 'dir'); mkdir(RESULTS_DIR); end

% -------------------- DISPATCH TO DATASET SCRIPT --------------------
runner_name = ['run_' DATASET];

fprintf("=== FeatureSelectionKLSR | dataset=%s ===\n", DATASET);
fprintf("CFG: N=%d | MaxIt=%d | runs=%d | test_ratio=%.2f | seed0=%d\n", ...
    CFG.N, CFG.MaxIt, CFG.runs, CFG.test_ratio, CFG.seed0);

if exist(runner_name, 'file') ~= 2
    error("Missing dataset runner: %s.m (put it under run/datasets/)", runner_name);
end

% Each dataset runner should have signature:
%   run_<dataset>(CFG)
feval(runner_name, CFG);

fprintf("=== DONE: %s ===\n", DATASET);
