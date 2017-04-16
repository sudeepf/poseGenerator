path = ['/home/capstone/Sudeep/Capstone/deep-3DHyPE/Dataset_2d/'];
vPath = [path,'/images/'];
pPath = [path,'/mpii_human_pose_v1_u12_1.mat'];
fNames = dir(vPath);
load(pPath);
imgs = [];
k=1;
poses2 = [];
for i=1:length(RELEASE.annolist)
	if exist([vPath, RELEASE.annolist(i).image.name], 'file') ~= 2
		continue
	end
	disp([vPath, RELEASE.annolist(i).image.name])
	%image = imread([vPath, RELEASE.annolist(i).image.name]);
	%imshow(image)
    %hold on
    
	if isfield(RELEASE.annolist(i).annorect, 'annopoints') == 0
        continue
    end
    for p=1:length(RELEASE.annolist(i).annorect)
        joints_mpi = ones(16,2) * -1000;
        if isfield(RELEASE.annolist(i).annorect(p).annopoints, 'point') == 0
            continue
        end
        for j=1:length(RELEASE.annolist(i).annorect(p).annopoints.point)
            joints_mpi(RELEASE.annolist(i).annorect(p).annopoints.point(j).id + 1,1) ...
             = RELEASE.annolist(i).annorect(p).annopoints.point(j).x ;
            joints_mpi(RELEASE.annolist(i).annorect(p).annopoints.point(j).id + 1,2) ...
             = RELEASE.annolist(i).annorect(p).annopoints.point(j).y ; 
        end
        joints_hum3 = ones(14,2) * -1000;
        %% Here We change the index
        joints_hum3(1,:) = joints_mpi(10,:);
        joints_hum3(2,:) = joints_mpi(9,:);
        joints_hum3(3,:) = joints_mpi(13,:);
        joints_hum3(4,:) = joints_mpi(12,:);
        joints_hum3(5,:) = joints_mpi(11,:);
        joints_hum3(6,:) = joints_mpi(14,:);
        joints_hum3(7,:) = joints_mpi(15,:);
        joints_hum3(8,:) = joints_mpi(16,:);
        joints_hum3(9,:) = joints_mpi(3,:);
        joints_hum3(10,:) = joints_mpi(2,:);
        joints_hum3(11,:) = joints_mpi(1,:);
        joints_hum3(12,:) = joints_mpi(4,:);
        joints_hum3(13,:) = joints_mpi(5,:);
        joints_hum3(14,:) = joints_mpi(6,:);

        %%
        
        poses2(k,:,:) = joints_hum3;
        k = k+1;
        imgs = [imgs; [vPath, RELEASE.annolist(i).image.name]];    
    end
    
end
save([path,'2DPose.mat'],'imgs','poses2')
