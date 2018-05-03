load('ORL_64x64.mat');
[nSmp,nFea] = size(fea);
for i = 1:nSmp
     fea(i,:) = fea(i,:) ./ max(1e-12,norm(fea(i,:)));
end
%fea = fea(1:100,:);
pic_num = 10;
train_num = 9;
train_person = 10;
fea_train = [];   %ÿ������10��ͼƬ��ȡǰ9����Ϊѵ����������1������
for i = 0:train_person-1      %ȡ10���˵���Ƭ�����Թ���90����Ƭ?
	fea_train = [fea_train fea(i*train_person+1 : i*train_person+9,:)'];
end
Sum = sum(fea_train,2);      
ave = Sum ./ size(fea_train,2);  %����ƽ����?
A = [];              %����A����
for i = 1:size(fea_train,2)
    E = (fea_train(:,i) - ave);
	A = [A E];
end
L = A'*A;
[V, D] = eig(L);
mu = A * V;                   %������������
mu = fliplr(mu);
FeatureVectors = mu(:,1:16);  %ȡǰ16��������

faceW = 64; 
faceH = 64; 
numPerLine = 4; 
ShowLine = 4; 

Y = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(FeatureVectors(:,i*numPerLine+j+1),[faceH,faceW]); 
  	end 
end
figure(1)
imshow(Y,[]); title('eigenfaces');    %��ʾ������

Projectparam = FeatureVectors' * A;   %��������ռ�
classparam = [];
for i = 0:9:train_num*(train_person-1)
    classparam = [classparam (sum(Projectparam(:,i+1:i+9),2) ./ train_num)];
end

clsthreshold = [];
for i = 1:train_person
	for j = 1:9
		clsthreshold = [clsthreshold sqrt(sum((Projectparam(:,(i-1)*train_num + j) - classparam(:,i)).^2))];
	end
end

%reconstruct = FeatureVectors * Projectparam;   %�ع�ͼ��

score = 0;     %����ʶ����
k = 0;
cls_score = [];
for j = 1:train_person*pic_num
	Testimage = fea(j,:)';
	%imshow(reshape(Testimage, 64, 64),[]);
	distance = FeatureVectors' * (Testimage - ave);
	cls_d = [];
	for i = 1:10
		cls_d = [cls_d sqrt(sum((distance - classparam(:,i)).^2))];   %����ͼƬ��ÿһ��ľ���?
	end
	[minimum, index] = min(cls_d);
	if index == gnd(j,1)
		score = score + 1;
        k = k + 1;
    end
    if mod(j,10)==0
        cls_score = [cls_score k/pic_num];
        k = 0;
    end
end 
figure(2)
bar(cls_score); title('Class accuracy'); xlabel('Class');ylabel('Accuracy'); axis([0 11 0 1]);
Accuracy = score/(train_person*pic_num); 
disp(['Whole accuracy:', num2str(Accuracy)])