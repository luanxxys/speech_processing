function show(type)
{
    count = 0;
    for (var key in methods) {
        var row = document.getElementById(key);
        if ((methods[key] &  type) != 0) {
            row.style.display = '';
            row.className = (count++ % 2) ? rowColor : altColor;
        }
        else
            row.style.display = 'none';
    }
    updateTabs(type);
}

function updateTabs(type)
{
    for (var value in tabs) {
        var sNode = document.getElementById(tabs[value][0]);
        var spanNode = sNode.firstChild;
        if (value == type) {
            sNode.className = activeTableTab;
            spanNode.innerHTML = tabs[value][1];
        }
        else {
            sNode.className = tableTab;
            spanNode.innerHTML = "<a href=\"javascript:show("+ value + ");\">" + tabs[value][1] + "</a>";
        }
    }
}


/************************************************************************/
function isEmpty(strValue){
	return null==strValue || 0>=strValue.length();
}

var gIsIfly=true;

function gIsIflytek(){
	return gIsIfly;
}

// 作者
var gIflyAuthorName = "iFlytek";
var gIflyAuthorLink = "http://www.xfyun.cn";
var gIflyAuthorLinkName = "讯飞语音云";

var gLxyAuthorName = "CMCC";
var gLxyAuthorLink = "http://dev.10086.cn";
var gLxyAuthorLinkName = "中国移动灵犀云";
 
function gShowAuthor(){
	var name = gIflyAuthorName;
	var link = gIflyAuthorLink;
	var linkName = gIflyAuthorLinkName;
	
	if(  !gIsIflytek() ){
		name = gLxyAuthorName;
		link = gLxyAuthorLink;
		linkName = gLxyAuthorLinkName;
	}
	
     document.write( name+"&nbsp;&nbsp;&nbsp;<a href="+link+">"+linkName+"</a>" );
}

// 语法指南
var gIflyGrammarGuideName = "语法编写指南";
var gIflyGrammarGuideLink = "http://club.voicecloud.cn/forum.php?mod=viewthread&tid=7595";

var gLxyGrammarGuideName = "语法编写指南";
var gLxyGrammarGuideLink = "http://dev.10086.cn/cmdn/bbs/thread-97556-1-1.html ";	

function gShowGrammarGuide(){
	var name = gIflyGrammarGuideName;
	var link = gIflyGrammarGuideLink;
	if( !gIsIflytek() ){
		name = gLxyGrammarGuideName;
		link = gLxyGrammarGuideLink;
	}
	
	document.write( "<p>关于语法文件的编写，请参考 <a href="+link+">"+name+"</a></p>" );
}

//开发者网站
var gIflyDeveloperWebsiteLink = "http://open.voicecloud.cn/developer.php";
var gIflyDeveloperWebsiteName = gIflyDeveloperWebsiteLink;

var gLxyDeveloperWebsiteLink = "http://dev.10086.cn"; 
var gLxyDeveloperWebsiteName = gLxyDeveloperWebsiteLink;

function gShowDeveloperWebsite(){
	var name = gIflyDeveloperWebsiteName;
	var link = gIflyDeveloperWebsiteLink;
	if( !gIsIflytek() ){
		name = gLxyDeveloperWebsiteName;
		link = gLxyDeveloperWebsiteLink;
	}
	document.write( "<a href="+link+">"+name+"</a>" );
}

// Java 常见问题
var gIflyJavaQnAName = "Java 常见问题解答"; 
var gIflyJavaQnALink = "http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=9864&pid=43664";

var gLxyJavaQnAName = "Java 常见问题解答";
var gLxyJavaQnALink = "http://dev.10086.cn/cmdn/bbs/viewthread.php?tid=97561";

function gShowJavaQnA(){
	var name = gIflyJavaQnAName;
	var link = gIflyJavaQnALink;
	if( !gIsIflytek() ){
		name = gLxyJavaQnAName;
		link = gLxyJavaQnALink;
	}
	
	document.write( "<a href="+link+">"+name+"</a>" );
}

//语义配置
var gIflySchWebsiteName = "语义开放平台";
var gIflySchWebsiteLink = "http://osp.voicecloud.cn";

var gLxySupportEmail = "lingxicloud@139.com";

function gShowSchApply(){
	if( gIsIflytek() ){
		document.write( "申请SDK时，默认未开通语义功能，如需使用请在<a href="+gIflySchWebsiteLink+">"+gIflySchWebsiteName+"</a>上进行业务配置" );
	}else{
		document.write( "申请SDK时，默认未开通语义功能，如需使用请联系 "+gLxySupportEmail+" 申请" );
	}
}

//服务器地址
var gIflyServerurlLink = "http://www.voicecloud.cn/msp.do";
var gLxyServerLink = "http://cmcc.lingxicloud.com/msp.do";
function gShowServerurl(){
	var link = gIflyServerurlLink;
	if( !gIsIflytek() ){
		link = gLxyServerLink;
	}
	document.write( link );
}

//页底文字
var gIflyBottom = "IFLYTEK";
var gLxyBottom = "CMCC";
function gShowBottom(){
	var bottom = gIflyBottom;
	if( !gIsIflytek() ){
		bottom = gLxyBottom;
	}
	document.write( "<center>"+bottom+"</center>" );
}

/************************************************************************/