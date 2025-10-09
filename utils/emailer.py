import smtplib
from email.mime.text import MIMEText
from utils import Config


def send_email(subject, content):
    mail_host = Config.SENDER_HOST
    sender = Config.SENDER
    mail_pw = Config.SENDER_PW
    receiver = Config.RECEIVER

    # Create the container (outer) email message.
    msg = MIMEText(content, "plain", "utf-8")
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        smtp = smtplib.SMTP_SSL(mail_host, 465)  # 实例化smtp服务器
        smtp.login(sender, mail_pw)  # 登录
        smtp.sendmail(sender, receiver, msg.as_string())
        print("Email send successfully")
    except smtplib.SMTPException:
        print("Error: email send failed")



if __name__ == "__main__":
    name = 'model'
    epoch = 100
    total_time = 100000
    train_epoch_loss = 0.1
    subject = name + " training is complete!!!"
    epoch_content ='total_epoch:' + str(epoch) + '\ntotal_time:' + str(total_time) + '\n'
    loss_content = 'last_train_loss:' + str(train_epoch_loss)
    content = subject +'\n' + epoch_content + loss_content
    send_email(subject, content)