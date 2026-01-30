package com.logis.scanner.v110

import android.os.Bundle
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.view.ViewGroup
import androidx.activity.enableEdgeToEdge

class MainActivity : TauriActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        
        // WebView를 찾아서 카메라 권한 요청을 자동으로 승인하는 WebChromeClient 설정
        window.decorView.post {
            findWebView(window.decorView as ViewGroup)?.let { webView ->
                webView.webChromeClient = object : WebChromeClient() {
                    override fun onPermissionRequest(request: PermissionRequest) {
                        // 카메라 및 오디오 권한 요청 시 즉시 승인
                        request.grant(request.resources)
                    }
                }
            }
        }
    }

    private fun findWebView(group: ViewGroup): WebView? {
        for (i in 0 until group.childCount) {
            val child = group.getChildAt(i)
            if (child is WebView) return child
            if (child is ViewGroup) {
                val found = findWebView(child)
                if (found != null) return found
            }
        }
        return null
    }
}
