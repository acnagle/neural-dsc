% In this MATLAB script, we evaulate the non-learning baseline on random Bernoulli messages (iid). This script requires the communications toolbox

clear all
close all
rng(1234)

P = [16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
     25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
     25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
      9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
     24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
      2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
    ];
blockSize = 27;
pcmatrix = ldpcQuasiCyclicMatrix(blockSize,P);


pcmatrix_org = pcmatrix; 
H = pcmatrix; 


rows = size(H, 1);
cols = size(H, 2);

r = 1;
for c = cols - rows + 1:cols
    if H(r,c) == 0
        % Swap needed
        for r2 = r + 1:rows
            if H(r2,c) ~= 0
                tmp = H(r, :);
                H(r, :) = H(r2, :);
                H(r2, :) = tmp;
            end
        end
    end

    % Ups...
    if H(r,c) == 0
        error('H is singular');
    end

    % Forward substitute
    for r2 = r + 1:rows
        if H(r2, c) == 1
            H(r2, :) = xor(H(r2, :), H(r, :));
        end
    end

    % Back Substitution
    for r2 = 1:r - 1
        if H(r2, c) == 1
            H(r2, :) = xor(H(r2, :), H(r, :));
        end
    end

    % Next row
    r = r + 1;
end

pcmatrix = H; %% Already standardized! 

cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);
cfgLDPCDec = ldpcDecoderConfig(pcmatrix);

% Evaluate on iid Bernoulli data
alpha_arr = linspace(0, 0.03, 51);     % probability of bit flip of side information
ber_arr = zeros(size(alpha_arr));
bler_arr = zeros(size(alpha_arr));

n_samples = 1000;
bern_data = randi([0 1], 648, n_samples);%readmatrix('./baseline_data/test_set.csv');

% Create generator and parity check matrices
[data_dim, num_samples] = size(bern_data);

m = 648; %ceil(log2(data_dim));       % Compute the minimum number of parity bits required for the data set
k = 162; % n - m;                      % message length

H = pcmatrix; % 162 by 648. 
Ppart = H(:,1:m-k); % 162 by 486. 

x = bern_data;

s = mod(H*x, 2);                                     % syndromes s^(n-k) = x^n * H'
e = [zeros(m-k,num_samples);s];
v = mod(x + e, 2);                                      % use syndromes to create valid codewords. v^n = x^n + e^n
assert(sum(mod(H*v, 2), 'all') == 0)                 % ensure that v contains valid codewords
maxnumiter = 6; 

for i=1:length(alpha_arr)
    alpha = alpha_arr(i);
    y = mod(v + floor(rand(m, num_samples) + alpha), 2); % add noise to the data to construct the side information

    % Decode to obtain the messages
    b_hat = ldpcDecode(-2*y+1,cfgLDPCDec,maxnumiter); %decode(y, m, k, 'hamming/binary', prim_poly);   % b_hat^k = decode(y)
    
    v_hat = mod(ldpcEncode(b_hat, cfgLDPCEnc), 2);                              % v_hat^n = b_hat^k * G
    x_hat = mod(double(v_hat) + e, 2);                              % x_hat^n = v_hat^n + e^n
    ber_arr(i) = sum(mod(x_hat + x, 2), 'all') / (n_samples * size(x_hat, 1));  % average fraction of mistakes
    bler_arr(i) = sum(sum(mod(x_hat + x, 2))~=0)/ n_samples; 
end
ber_arr
bler_arr

f = figure;
plot(alpha_arr, ber_arr,'b'); hold on; 
plot(alpha_arr, alpha_arr,'k'); hold on; % side information 
plot(alpha_arr, 0.75*alpha_arr,'r'); % no side info
title('Bernoulli IID Non-learning Baseline - Bit error rate')
xlabel('Probability of a Bit Flip')
ylabel('BER')

figure; 
plot(alpha_arr, bler_arr,'b'); hold on; %% DISCUS 
plot(alpha_arr, 1-(1-alpha_arr).^m,'k'); hold on; % Use side-info only ; let xhat = y (then BLER is given as 1-(1-alpha_arr).^m)
plot(alpha_arr, 1-(1-0.5).^(m-k)*ones(size(alpha_arr)),'r') % Use compressed message only; k bits are correctly transmitted.We do random guessing on (m-k) remaining bits. BER is 1. 
title('Block error rate')

Flip = alpha_arr; 
LDPC_err = bler_arr; 
SIDEINFO_only = 1-(1-alpha_arr).^m; 
COMPRESSEDMESSAGE_only = 1-(1-0.5).^(m-k)*ones(size(alpha_arr)); 

dlmwrite('iid_flip.csv',Flip,'-append','delimiter',',')
dlmwrite('iid_flip.csv',LDPC_err,'-append','delimiter',',')
dlmwrite('iid_flip.csv',SIDEINFO_only,'-append','delimiter',',')
dlmwrite('iid_flip.csv',COMPRESSEDMESSAGE_only,'-append','delimiter',',')

