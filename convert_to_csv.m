clear; clc;

% Dataset folder
folder = 'C:\Users\vishn\Downloads\5. Battery Data Set';

if ~isfolder(folder)
    error('Dataset folder not found. Check the folder path.');
end

files = dir(fullfile(folder, '**', '*.mat'));

if isempty(files)
    error('No .mat files found in the dataset folder.');
end

% Output tables
summary = table();
measurements = table();

for f = 1:numel(files)

    filepath = fullfile(files(f).folder, files(f).name);
    fprintf('Processing: %s\n', files(f).name);

    S = load(filepath);
    names = fieldnames(S);

    % Find battery structure
    B = [];
    for k = 1:numel(names)
        x = S.(names{k});
        if isstruct(x) && isfield(x, 'cycle')
            B = x;
            break;
        end
    end

    if isempty(B)
        warning('No cycle structure found in %s', files(f).name);
        continue;
    end

    battery_id = erase(files(f).name, '.mat');
    cycles = B.cycle;

    for c = 1:numel(cycles)

        cyc = cycles(c);

        if ~isfield(cyc, 'type') || ...
                ~strcmpi(strtrim(char(string(cyc.type))), 'discharge')
            continue;
        end

        if ~isfield(cyc, 'data') || ~isstruct(cyc.data)
            continue;
        end

        d = cyc.data;

        % Extract available signals
        voltage = getVector(d, 'Voltage_measured');
        current = getVector(d, 'Current_measured');
        temp = getVector(d, 'Temperature_measured');
        current_load = getVector(d, 'Current_load');
        voltage_load = getVector(d, 'Voltage_load');
        time = getVector(d, 'Time');

        capacity = getScalar(d, 'Capacity');

        ambient_temp = NaN;
        if isfield(cyc, 'ambient_temperature')
            ambient_temp = getScalar(cyc, 'ambient_temperature');
        end

        n = max([numel(voltage), numel(current), ...
            numel(temp), numel(current_load), ...
            numel(voltage_load), numel(time)]);

        if n == 0
            warning('No measurements in %s, cycle %d', battery_id, c);
            continue;
        end

        % One row per discharge cycle
        duration = NaN;
        if numel(time) >= 2
            duration = time(end) - time(1);
        end

        row = table( ...
            string(battery_id), c, ambient_temp, capacity, n, duration, ...
            meanOrNaN(voltage), minOrNaN(voltage), maxOrNaN(voltage), ...
            meanOrNaN(current), minOrNaN(current), maxOrNaN(current), ...
            meanOrNaN(temp), minOrNaN(temp), maxOrNaN(temp), ...
            'VariableNames', { ...
            'battery_id','cycle','ambient_temp','capacity', ...
            'n_measurements','duration', ...
            'voltage_mean','voltage_min','voltage_max', ...
            'current_mean','current_min','current_max', ...
            'temp_mean','temp_min','temp_max'});

        summary = [summary; row];

        % One row per individual measurement
        idx = (1:n)';

        rows = table( ...
            repmat(string(battery_id), n, 1), ...
            repmat(c, n, 1), ...
            repmat(capacity, n, 1), ...
            idx, ...
            padVector(time, n), ...
            padVector(voltage, n), ...
            padVector(current, n), ...
            padVector(temp, n), ...
            padVector(current_load, n), ...
            padVector(voltage_load, n), ...
            'VariableNames', { ...
            'battery_id','cycle','capacity','measurement', ...
            'time','voltage','current','temp', ...
            'current_load','voltage_load'});

        measurements = [measurements; rows];
    end
end

if isempty(summary)
    error(['No discharge cycles extracted. Check the MAT-file ', ...
           'structure and cycle labels.']);
end

% Export CSV files
writetable(summary, fullfile(folder, ...
    'battery_cycle_summary.csv'));

writetable(measurements, fullfile(folder, ...
    'battery_discharge_measurements.csv'));

fprintf('\nConversion complete!\n');
fprintf('Discharge cycles: %d\n', height(summary));
fprintf('Measurement rows: %d\n', height(measurements));
fprintf('Output folder: %s\n', folder);

%% Helper functions

function x = getVector(s, name)
    x = [];
    if isfield(s, name)
        v = s.(name);
        if isnumeric(v) || islogical(v)
            x = double(v(:));
        end
    end
end

function x = getScalar(s, name)
    x = NaN;
    if isfield(s, name)
        v = s.(name);
        if isnumeric(v) && ~isempty(v)
            x = double(v(1));
        end
    end
end

function x = padVector(v, n)
    x = NaN(n, 1);
    m = min(numel(v), n);
    if m > 0
        x(1:m) = v(1:m);
    end
end

function x = meanOrNaN(v)
    if isempty(v)
        x = NaN;
    else
        x = mean(v, 'omitnan');
    end
end

function x = minOrNaN(v)
    if isempty(v)
        x = NaN;
    else
        x = min(v, [], 'omitnan');
    end
end

function x = maxOrNaN(v)
    if isempty(v)
        x = NaN;
    else
        x = max(v, [], 'omitnan');
    end
end