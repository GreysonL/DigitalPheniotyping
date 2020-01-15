ids = list.files("F:/DATA/hope")
for(id in ids){
  name = list.files(paste0("F:/DATA/hope/",id,"/identifiers"))[1]
  file = read.csv(paste0("F:/DATA/hope/",id,"/identifiers/",name),header=T)
  if (as.numeric(rownames(file))>1){
    temp = as.matrix(file)[1,]
    temp = c(row.names(file)[1],as.vector(temp[-13]))
    row.names(file) = NULL
    temp = as.data.frame(t(temp))
    colnames(temp) = colnames(file)
    write.csv(temp,paste0("F:/DATA/hope/",id,"/identifiers/",name),row.names = F)
  }
}