# coding=utf-8
"""邮件操作函数"""
import os
import email.mime.text
import email.mime.base
import email.mime.multipart
import email.encoders
import email.utils
from email.header import decode_header
from email.parser import BytesParser
import smtplib 
import mimetypes
import poplib

# 登录邮箱
def loginEmailServer(server,protocol,address,password,ssl=True):
    if protocol in ["pop", "pop3"]:
        if ssl:
            EmailServer = poplib.POP3_SSL(server, 995)
        else:
            EmailServer = poplib.POP3(server, 110)
    elif protocol in ["imap", "imap4"]:
        if ssl:
            EmailServer = poplib.POP3_SSL(server, 995)
        else:
            EmailServer = poplib.POP3(server, 110)
    else:
        return (-1,"Parse protocol failed: {0}".format(protocol))
    #print(EmailServer.getwelcome())# debug
    try:
        EmailServer.user(address)
        EmailServer.pass_(password)
    except:
        return (-1,"Unable to log on!")
    return (1,EmailServer)
# 获取邮件的字符编码，首先在message中寻找编码，如果没有，就在header的Content-Type中寻找
def guessMailCharset(msg):
    Charset = msg.get_charset()
    if Charset is None:
        ContentType = msg.get('Content-Type', '').lower()
        Pos = ContentType.find('charset=')
        if Pos>=0:
            Charset = ContentType[Pos+8:].strip()
    return Charset
# 解析消息头中的字符串
def decodeHeaderStr(value):
    value, Charset = decode_header(value)[0]
    if Charset:
        value = value.decode(Charset)
    return value
# 下载邮件的附件
def downloadMailAccessory(msg,save_dir):
    FileList = []
    for iPart in msg.walk():
        iFileName = iPart.get_filename()
        if iFileName:
            iFileName = decodeHeaderStr(iFileName)
            iData = iPart.get_payload(decode=True)
            if (iFileName is not None) and (iFileName!=''):
                FileList.append(save_dir+os.sep+iFileName)
                iFile = open(FileList[-1], 'wb')
                iFile.write(iData)
                iFile.close()
    return FileList
# 获取邮件的头
def getMailHeaderInfo(msg):
    HeaderInfo = {}
    for iHeader in ['From', 'To', 'Subject']:
        iValue = msg.get(iHeader, '')
        if iValue:
            if iHeader == 'Subject':
                iValue = decodeHeaderStr(iValue)
            else:
                iTemp, iAddr = email.utils.parseaddr(iValue)
                iName = decodeHeaderStr(iAddr)
                iValue = iName + ' < ' + iAddr + ' > '
            HeaderInfo[iHeader] = iValue
    return HeaderInfo
# 获取邮件的正文
def getMailContent(msg,content_file_name=None,save_dir=None):
    MailContent = {}
    for iPart in msg.walk():
        iFileName = iPart.get_filename()
        iContentType = iPart.get_content_type()
        iCharset = guessMailCharset(iPart)
        if iFileName:
            continue
        else:
            if iContentType == 'text/plain':
                MailContent['ContentType'] = 'text'
            elif iContentType == 'text/html':
                MailContent['ContentType'] = 'html'
            if iCharset:
                MailContent['Content'] = iPart.get_payload(decode=True).decode(iCharset)
            else:
                MailContent['Content'] = iPart.get_payload(decode=True)
            if save_dir is not None:
                if content_file_name is None:
                    if MailContent['ContentType']=='html':
                        content_file_name = genAvailableName('Mail',listDirFile(save_dir,'html'))+'.html'
                    else: 
                        content_file_name = genAvailableName('Mail',listDirFile(save_dir,'txt'))+'.txt'
                iFile = open(save_dir+os.sep+content_file_name, 'w')
                iFile.write(MailContent['Content'])
                iFile.close()
    return MailContent
# 解析邮件
def parseMail(resp, lines, octets):
    Msg = '\r\n'.encode().join(lines)
    return BytesParser().parsebytes(Msg)
# 发送邮件
def sendMail(mail_addr,subject,file_path,password):
    #命令 mail.py <1:发送方（回复地址）10000@qq.com> <2:发送地址，多个以;隔开> <3:发送文件>  
    From = "MultiFactor<MultiFactor@163.com>"
    #ReplyTo='MultiFactor@163.com'
    To = mail_addr
    server = smtplib.SMTP("smtp.163.com",25)
    server.login("MultiFactor@163.com",password)#仅smtp服务器需要验证时
    # 构造MIMEMultipart对象做为根容器
    main_msg = email.mime.multipart.MIMEMultipart()
    # 构造MIMEText对象做为邮件显示内容并附加到根容器
    text_msg = email.mime.text.MIMEText("")
    main_msg.attach(text_msg)
    # 构造MIMEBase对象做为文件附件内容并附加到根容器  
    ctype,encoding = mimetypes.guess_type(file_path)  
    if (ctype is None) or (encoding is not None):  
        ctype='application/octet-stream'  
    maintype,subtype = ctype.split('/',1)
    file_msg = email.mime.base.MIMEBase(maintype,subtype)
    File = open(file_path,'rb')
    file_msg.set_payload(File.read())
    File.close()
    email.encoders.encode_base64(file_msg)
    # 设置附件头  
    basename = os.path.basename(file_path)  
    file_msg.add_header('Content-Disposition','attachment', filename=basename)#修改邮件头  
    main_msg.attach(file_msg)
    # 设置根容器属性  
    main_msg['From'] = From
    main_msg['To'] = To
    main_msg['Subject'] = subject
    main_msg['Date'] = email.utils.formatdate()
    # 用smtp发送邮件  
    try:  
        server.sendmail(From, To.split(';'), main_msg.as_string())
        #server.send_message(main_msg,From,to_addrs=ReplyTo)
    finally:  
        server.quit()
# 获取邮件数据补丁包的日期序列
def getPatchDate(subject_header="DSData"):
    ErrorCode,MailServer = loginEmailServer(**{"server":"pop.163.com","protocol":"pop3","ssl":True,"address":"MultiFactor@163.com","password":"jhqbbrnerwoytrrs"})
    if ErrorCode==-1:
        return (-1,MailServer)
    nMsg = len(MailServer.list()[1])
    PatchDates = []
    nSubjectHeader = len(subject_header)
    for i in range(1,nMsg+1,1):
        iSubject = MailServer.top(i,0)[1][7].decode("utf-8").split(": ")[1]
        if (iSubject is not None) and (iSubject[:nSubjectHeader]==subject_header):
            iDate = iSubject[nSubjectHeader:]
            PatchDates.append(iDate)
    MailServer.quit()
    return (1,PatchDates)