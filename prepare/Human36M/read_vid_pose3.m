% Read imagefiles and 3D pose

subjects = [  'S0'];
% Dataset Paths 
for iok=1:length(subjects)
    subject = subjects(iok:iok+1);
    path = ['/home/capstone/Sudeep/Capstone/3Dpose/Dataset/',subject];
    vPath = [path,'/Videos/'];
    pPath = [path,'/Pose/D3_Positions_mono/'];
    p2Path = [path,'/Pose/D2_Positions/'];
   
    pFiles = dir([pPath,'*cdf']);
    p2Files = dir([p2Path,'*cdf']);
    
    vFolders = dir([vPath]);
    vFolders(1:2) = [];
    dirFlag = [vFolders.isdir];
    vFolders = vFolders(dirFlag);

    system(['mkdir ', path,'/mats']);
    matPath = [path,'/mats'];
    %% Looping over all the video folders
    for ii=1:length(vFolders)
        fName = vFolders(ii).name;
        fName1 = strcat(vPath,fName);
        iName = dir([fName1,'/*jpg']);
        disp([pPath,fName,'.cdf']);
        p3D = cdfread([pPath,fName,'.cdf']);
        p3D = p3D{1};
        p2D = cdfread([p2Path,fName,'.cdf']);
        p2D = p2D{1};
        imgs = [];
        poses2 = [];
        poses3 = [];
        p3GT = [];
        camC = [];
        matName = [matPath,'/',fName,'.mat'];
        %% Looping over all the images and pose files 
        for j = 1:1:length(p3D)
            imN = iName(j).name;
            %im = imread([vPath,fName,'/',imN]);
            p3d = p3D(j,:);
            p2d = p2D(j,:);
            imgs = [imgs; strcat(vPath,'/',fName,'/',imN)];
            % Extracing pose from strange data type
            pose3 = extract3DJoints(p3d);
            p3 = pose3;
            pose2 = extract2DJoints(p2d);
          
            %Centering the pose2d 
            %[H W C] = size(im);
            %Computing Camera Params
            base2_1 = pose2(9,:);
            base2_2 = pose2(12,:);
            base3_1 = pose3(9,1:2)./pose3(9,3);
            base3_2 = pose3(12,1:2)./pose3(12,3);
            
             
            
            nume_ = base2_1.*base3_2 - base2_2.*base3_1;
            denom_ = base3_2 - base3_1;
            F = nume_./denom_; % Camera parameters 
            %midP = base2_1 - (base3_1).*F ;
            midP = nume_./denom_;
            pose2_ = repmat(midP,14,1) - pose2;
            

            % Centering 3D pose wrt z axis

            %Computing Scale of the projection
            ProjScale = pose3(:,1:2) ./ -pose2_;

    %%         Visualization


            %pose3(:,3) = pose3(:,3) - repmat(m3_root(1,3),14,1);
            %pose3(:,1:2) = pose3(:,1:2) + repmat(midP,14,1);
            mean_proj = mean(mean(ProjScale));
            pose3 = [pose3(:,1)./mean_proj,pose3(:,2)./mean_proj,pose3(:,3)./mean_proj];
            %pose3 = [pose3(:,1)./ProjScale(:,1),pose3(:,2)./ProjScale(:,2),pose3(:,3)./mean(mean(ProjScale))];
            
            poses2(j,:,:) = pose2;
            m3_root = 0.5*([pose3(9, 1) pose3(9, 2) pose3(9,3)]+...
                [pose3(12,1) pose3(12,2) pose3(12,3)]);

            m2_root = 0.5*([pose2(9, 1) pose2(9, 2)]+...
                [pose2(12,1) pose2(12,2)]);

            pose3(:,3) = pose3(:,3) - repmat(m3_root(1,3),14,1);
            pose3(:,1:2) = pose3(:,1:2) + repmat(midP,14,1);
%             figure(2)
%             scatter3(pose3(:,1),pose3(:,2),pose3(:,3))
%              hold on,
%              scatter(pose2(:,1),pose2(:,2))
%              set(gca,'XLim',[0 1000],'YLim',[0 1000],'ZLim',[-500 500])
%              figure(1), imshow(im)
%            pause
            poses3(j,:,:) = pose3;
            p3GT(j,:,:) = p3;
            camC(j,:) = midP; 
        end
        save(matName,'imgs','poses2','poses3', 'p3GT', 'camC')
        disp(['saved',matName])
    end
end
length(vFolders)
