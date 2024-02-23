function [B] = alias_removal(img)
%FIXME : hanning window
%FIXME : use zeros in the FFT domain near 0 frequency + add butterworth filter (butter on matlab)

% img is a gray image for now (2D-matrix)

% img_bin = imbinarize(img, 0.33);

% img = flat_field_correction(double(img), 50);

%% parameters
blur = 5;
threshold = 0.18;
num_peaks = 4;

% img_bin = gpuArray(img_bin);
img = gpuArray(img);
img = single(img);
img_org = img;
img = img - mean(img,"all");

Ny = size(img, 1);
Nx = size(img, 2);

% %% FFT section
% fft_img = fftshift(fft2(img));
% figure(1)
% imagesc(log10(abs(fft_img)));
% fft_img_filtered = zeros(size(fft_img));
% fft_img_filtered(100:end-100,100:end-100) = fft_img(100:end-100,100:end-100);
% %ring test
% border1 = 150;
% border2 = 200;
% mask = ones(size(fft_img));
% mask(border1:end-border1,border1:end-border1) = zeros(size(fft_img,1)-2*border1+1);
% mask(border2:end-border2,border2:end-border2) = ones(size(fft_img,1)-2*border2+1);
% fft_img_filtered_ring = fft_img.*mask;
% figure(2)
% imagesc(mask)
% 
% 
% figure(3)
% imagesc(log10(abs(fft_img_filtered_ring)));
% 
% img_filtered = abs(ifft2(fftshift(fft_img_filtered_ring)));
% figure(4)
% imagesc(mat2gray(img_filtered));
% axis off
% axis image
% colormap gray
% colorbar
% 
% figure(5) % original img
% imagesc(mat2gray(img_org));
% axis off
% axis image
% colormap gray
% colorbar

figure(100)
imagesc(img)
axis square
axis off
colormap gray


%% FIND LOCATION OF ALIASES
% c = xcorr2(img, img_bin);
c = xcorr2(img);

% new_c = c(center_x - 100 : center_x + 100, :);
c = mat2gray(gather(c));
c = flat_field_correction(c, 30);

% [Z, R] = radialavg(c, 1000);

figure(1)
imagesc(c)
axis image
axis off

threshold = 0.71 * max(c, [], 'all');
c_bin = c > threshold;

figure(2)
imagesc(c_bin)
axis image
axis off

shifts = zeros(num_peaks, 2);
prom = zeros(num_peaks, 1);


for i = 1 : num_peaks
    L = bwareafilt(c_bin, 1, 'smallest');
    prom(i) = max(L .* c, [], "all");
    L_blurred = imgaussfilt(double(L), 5);
    [~, locs] = max(L_blurred, [], 'all');
    ind = [locs(1)];
    sz = [size(c, 1) size(c, 2)];
    [row,col] = ind2sub(sz,ind);
    shifts(i, :) = [(Ny + row), (Nx + col)];
    c_bin = logical(c_bin - L);
end


% figure(2)
% c_lineshape = mean(c(1022:1024,:),1);
% % c_lineshape = mean(c(:,:),1);
% plot(1:size(c,2),c_lineshape);
% %FIXME : fix the parameters in the findpeaks
% %FIXME : erease the peaks near the center of the lineshape
% [peaks,locs] = findpeaks(c_lineshape);
% disp(locs);
% disp(peaks);
% locs = [locs(2),locs(10)];
% pks = [peaks(2),peaks(10)];
% locs = [locs(4),locs(8)];
% pks = [peaks(4),peaks(8)];

%% FIND PERIODICITY
% periodicity = floor((locs(2) - locs(1))/2);
% % periodicity = 372;
% 
% shifted_img_x = zeros(size(img_org));
% shifted_img_x(:, 1:periodicity) = img_org(:, end - periodicity + 1 : end);
% shifted_img_x(:, end - periodicity + 1 : end) = img_org(:, 1:periodicity);
% % shifted_img_x(:, 1:periodicity) = img_org(:, end - periodicity + 1 : end);
% 
% figure(12)
% imagesc(shifted_img_x)
% axis off
% axis square
% colormap gray
gray_level = mean(img_org, "all");

%% SVD FILTERING
reshape(img_org,[size(img_org,1)*size(img_org,2),1]);
A = zeros(size(img_org,1)*size(img_org,2), num_peaks + 1);
AB = zeros(size(img_org,1),size(img_org,2), num_peaks);
blur_size = 100;
for ii=0
    
    tmp_img = ones(2*Ny, 2*Nx);
%     tmp_img = tmp_img .* gray_level;
    tmp_img(1:Ny, 1:Nx) = img_org;
    tmp_img = circshift(tmp_img, [floor(Ny/2) floor(Nx/2)]);

    apod_mask = zeros(2*Ny, 2*Nx);
    apod_mask(1+blur_size/2:Ny-blur_size/2, 1+blur_size/2:Nx-blur_size/2) = 1;
    apod_mask = circshift(apod_mask, [floor(Ny/2) floor(Nx/2)]);
    apod_mask = movmean(apod_mask, blur_size , 1);
    apod_mask = movmean(apod_mask, blur_size , 2);
    tmp_img = tmp_img .* apod_mask;
    tmp_img = tmp_img + gray_level.*(1-apod_mask);

    tmp_img_0 = circshift(tmp_img,[0,ii]);
    
    tmp_img_0 = tmp_img_0(floor(Ny/2) + 1 : floor(Ny/2) + Ny,  floor(Nx/2) + 1 : floor(Nx/2) + Nx);
    

    figure(11)
    for jj = 1 : num_peaks
        tmp_img_1 = circshift(tmp_img,[shifts(jj, 1), shifts(jj, 2)]);
        tmp_img_1 = tmp_img_1(floor(Ny/2) + 1 : floor(Ny/2) + Ny,  floor(Nx/2) + 1 : floor(Nx/2) + Nx);
        AB(:,:, jj + 1) = mat2gray(tmp_img_1);
        
        imagesc(tmp_img_1)
        axis square
        axis off
        colormap gray
        pause(1/10)
        hold on
        A(:,jj + 1) = reshape(tmp_img_1,[size(tmp_img_0,1)*size(tmp_img_0,2),1]);
    end

    A(:,1) = reshape(tmp_img_0,[size(tmp_img_0,1)*size(tmp_img_0,2),1]);


%     tmp_img_0 = circshift(img_org,[0,ii]);
%     tmp_img_1 = circshift(img_org,[0,locs(1)+ii]);
%     tmp_img_2 = circshift(img_org,[0,locs(2)+ii]);
%     tmp_img_0 = circshift(img_org,[ii, 0]);
%     tmp_img_1 = circshift(img_org,[locs(1)+ii, 0]);
%     tmp_img_2 = circshift(img_org,[locs(2)+ii, 0]);

    figure(111)
    subplot(1,5,1);
    imagesc(reshape(A(:,1),[size(tmp_img_0,1),size(tmp_img_0,2)]));
    axis square
    axis off
    colormap gray

    subplot(1,5,2);
    imagesc(reshape(A(:,2),[size(tmp_img_0,1),size(tmp_img_0,2)]));
    axis square
    axis off
    colormap gray

    subplot(1,5,3);
    imagesc(reshape(A(:,3),[size(tmp_img_0,1),size(tmp_img_0,2)]));
    axis square
    axis off
    colormap gray

    subplot(1,5,4);
    imagesc(reshape(A(:,4),[size(tmp_img_0,1),size(tmp_img_0,2)]));
    axis square
    axis off
    colormap gray

    subplot(1,5,5);
    imagesc(reshape(A(:,5),[size(tmp_img_0,1),size(tmp_img_0,2)]));
    axis square
    axis off
    colormap gray
% 
%     
%     A(:,2) = reshape(tmp_img_1,[size(tmp_img_1,1)*size(tmp_img_1,2),1]);
%     A(:,3) = reshape(tmp_img_2,[size(tmp_img_2,1)*size(tmp_img_2,2),1]);
end

% rgb_img = zeros(size(img_org, 1), size(img_org, 2), 3);
% rgb_img(:,:, 1) = mat2gray(tmp_img_0);
% rgb_img(:,:, 2) = mat2gray(tmp_img_1);
% rgb_img(:,:, 3) = mat2gray(tmp_img_2);
% 
% figure(22)
% imshow(rgb_img)

% A = zeros(size(img_org,1)*size(img_org,2),2);
% A(:,1) = reshape(img_org,[size(img_org,1)*size(img_org,2),1]);
% A(:,2) = reshape(shifted_img_x,[size(img_org,1)*size(img_org,2),1]);

[U,S,V] = svd(A,"econ");
size(S);

UU = reshape(U, size(img_org,1), size(img_org,2), []);

figure(112)
subplot(1,5,1);
imagesc(UU(:,:, 1));
axis square
axis off
colormap gray

subplot(1,5,2);
imagesc(UU(:,:, 2));
axis square
axis off
colormap gray

subplot(1,5,3);
imagesc(UU(:,:, 3));
axis square
axis off
colormap gray

subplot(1,5,4);
imagesc(UU(:,:, 4));
axis square
axis off
colormap gray
% 
subplot(1,5,5);
imagesc(UU(:,:, 5));
axis square
axis off
colormap gray

imwrite(mat2gray(UU(:, :, 1)), 'C:\\Users\Bronxville\Pictures\UU_1.png')

Threshold1 = 2;
Threshold2 = 5;
img_svded = U*S(:,Threshold1:Threshold2)*V(:,Threshold1:Threshold2)';
img_svded = reshape(img_svded,[size(img_org,1),size(img_org,2),size(S,1)]);

figure(113)
num_sub = Threshold2 - Threshold1 + 1;

for ii = 1: num_sub

subplot(1,num_sub,ii);
imagesc(img_svded(:,:, Threshold1 + ii - 1));
axis square
axis off
colormap gray

end


% figure(1)
% for ii=1:3
%     imagesc(img_svded(:,:,mod(ii,3)+1));
%     axis off
%     axis image
%     colormap gray
%     colorbar 
%     pause(0.3)
% end

pks = prom;
pks = pks - min(pks, [], 'all');
cor_factor = sqrt(mean(pks)/max(pks));



% cor_factor = mean(prom);
% cor_factor = 0.7;
% correction_img = mat2gray(UU(:,:, 2));
% img_to_remove = cor_factor*img_svded(:,:,2)+cor_factor*img_svded(:,:,3);
% img_to_remove = UU(:,:, 3);
img_to_remove = zeros(size(img_org));

for i = Threshold1 : Threshold2
    img_to_remove = img_to_remove + img_svded(:,:,i);
end


% for ff = 1 : length(tab_factor)
%     img_org - img_to_remove;
% end

% B = img_org-cor_factor*img_svded(:,:,2)-cor_factor*img_svded(:,:,3);
B = img_org-cor_factor*img_to_remove;
% img_to_remove = mat2gray(UU(:,:, 1));
% B = mat2gray(img_org) + 0.2.*img_to_remove;
% B = img_org-cor_factor*correction_img;

figure(7)
imagesc(mat2gray(img_to_remove));
axis off
axis image
colormap gray
colorbar

figure(8) % corrected img
imagesc(mat2gray(B));
axis off
axis image
colormap gray
colorbar

% imwrite(mat2gray(B), 'C:\Users\Bronxville\Downloads\230801_RLAB_0034_OS1_1_0_M0_cleaned.png')

figure(4) % original img
imagesc(mat2gray(img_org));
axis off
axis image
colormap gray
colorbar

end
