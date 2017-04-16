%% Readin Videos and Extract frames to a folder

subjects = ['S1'];

for ioo=1:2:length(subjects)
    subject = subjects(ioo:ioo+1);
    path = ['/home/capstone/datasets/Human3.6M/Subjects/',subject];
    vPath = [path,'/Videos/'];
    pPath = [path,'/Pose/D3_Positions_mono/'];

    pFiles = dir([pPath,'*cdf']);
    vFiles = dir([vPath, '*mp4']);

    for ii = 1:length(vFiles)
        fname = vFiles(ii).name;
        if(fname(1) == '_')
            continue,
        end
        ll = strsplit(fname,'.m')
        ll = ll(1)
        disp(strjoin(['mkdir ',vPath,ll],''))
        system(strjoin(['mkdir ',vPath,ll],''))
        ll = cell2mat(ll)
        lol = [vPath,ll,'/frame%6d.jpg']
        disp(['ffmpeg  -i ', [vPath,fname],'  ',  lol])
        system(['ffmpeg  -i ', [vPath,fname],'  ',  lol])
    end
end