if ~exist(['Return',PID],'var')
    eval(['Return',PID,'=[];']);
end
Return = eval(['Return',PID]);
Return = [Return;NewReturn];
[nAllDate,nID] = size(Return);
Return = Return(nAllDate-nDate+1:end,:);
TD = eye(nID);
for i=1:nID
    for j=i+1:nID
        TD(i,j) = TailDependence(Return(:,i),Return(:,j),'t','normal');
        TD(j,i) = TD(i,j);
    end
end
eval(['Return',PID,'=Return;']);