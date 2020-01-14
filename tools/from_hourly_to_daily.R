files = list.files("C:/Users/glius/Downloads/abdominal_stats")
for(i in files){
  data = read.csv(paste0("C:/Users/glius/Downloads/abdominal_stats/",i),header = T)
  s = 1
  for(k in 1:nrow(data)){
    if(data$hour[k]==0){s=s+1}
    data$key[k]=s
  }
  data = subset(data,key>1 & key<s)
  daily_stats = c()
  for(j in unique(data$key)){
    temp = subset(data,key==j)
    ht = sum(temp$home_time)
    dist = sum(temp$dist_traveled)
    obs = sum(60-temp$missing_time)
    daily_stats = rbind(daily_stats,c(temp$year[1],temp$month[1],temp$day[1],obs,ht/60,dist/1000))
  }
  id = strsplit(i, "_")[[1]][1]
  colnames(daily_stats)=c("year","month","day","obs_min","hometime_hr","dist_km")
  write.csv(daily_stats,paste0("C:/Users/glius/Desktop/output/",id,".csv"),row.names=FALSE)
}