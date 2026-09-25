export type Theme = "dark" | "light"

export const THEME_STORAGE_KEY = "sdc-theme"
export const DEFAULT_THEME: Theme = "dark"

export const themeScript = `(function(){try{var t=localStorage.getItem("${THEME_STORAGE_KEY}");if(t!=="light"&&t!=="dark"){t="${DEFAULT_THEME}"}var d=document.documentElement;d.classList.toggle("dark",t==="dark");d.style.colorScheme=t}catch(e){document.documentElement.classList.add("dark")}})()`
