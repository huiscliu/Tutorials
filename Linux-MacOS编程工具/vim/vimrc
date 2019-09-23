" Configuration file for vim

"General
set history=3000
filetype on
filetype plugin on
filetype plugin indent on
syntax on

"theme
"colorscheme 

" check one time after 4s of inactivity in normal mode
set autoread
au CursorHold * checktime

set hidden
set magic
set backspace=eol,start,indent
set incsearch
set hlsearch
set smartcase
set showmatch
set vb t_vb= 
set novisualbell
set tags=tags;

"For programming
set sm
set ru
set wildmenu
set wildmode=list:longest

"set autoindent
set smartindent
set ci 

set pastetoggle=<F2>
"set showmode
"set scrolloff=1000
"set mouse=r
set clipboard=unnamed
"set clipboard=unnamedplus
"set completeopt=longest,menu
"set synmaxcol=120
set ttyfast " u got a fast terminal
"set ttyscroll=3
set lazyredraw " to avoid scrolling problems

set encoding=utf-8
set fileencodings=utf-8,gb2312,gb18030,gbk,ucs-bom,cp936,latin1
set termencoding=utf-8

"deal with tab and shift
set smarttab
set et
set ts=4
set shiftwidth=4

"text width
set tw=110

" statusline
" cf the default statusline: %<%f\ %h%m%r%=%-14.(%l,%c%V%)\ %P
" format markers:
"   %< truncation point
"   %n buffer number
"   %f relative path to file
"   %m modified flag [+] (modified), [-] (unmodifiable) or nothing
"   %r readonly flag [RO]
"   %y filetype [ruby]
"   %= split point for left and right justification
"   %-35. width specification
"   %l current line number
"   %L number of lines in buffer
"   %c current column number
"   %V current virtual column number (-n), if different from %c
"   %P percentage through buffer
"   %) end of width specification
set laststatus=2
set statusline=%<\ %n:%f\ %m%r%y%=%-35.(%l\ of\ %L,\ %c%)
"au InsertEnter * set laststatus=0
"au InsertLeave * set laststatus=2


"File type fortran and c++
"High light
"au BufNewFile,BufRead *.{[f|F],[F|f][O|o][R|r],[f|F]90,[f|F]95,[f|F][p|P][p|P],[f|F]77,[f|F][T|t][n|N],fortran} setf fortran
au BufNewFile,BufRead *.{[c|C],[c|C][c|C],[c|C][p|P][p|P],[c|C][x|X][x|X],[i|I][p|P][p|P],[c|C]++,[t|T][c|C][c|C],moc,inl,[c|C][u|U]} setf cpp

au BufWinEnter *[mM]akefile*,*.mk,*.mak,*.dsp,Make*.in set noet
au BufWinEnter *[mM]akefile,*.mk,*.mak,*.dsp,Make*.in set noet
au BufWinEnter *[mM]akefile*,*.mk,*.mak,*.dsp,Make*.in set tw=0
au BufWinEnter *[mM]akefile,*.mk,*.mak,*.dsp,Make*.in set tw=0
au BufWinEnter *.m set tw=0
au BufWinEnter *[mM]akefile*,*.mk,*.mak,*.dsp,Make*.in set filetype=c

if has("autocmd")
  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif

" map keys
cnoremap <C-A>      <Home>
cnoremap <C-E>      <End>

map f <C-f>
map b <C-b>
map t <C-o>
map W <C-w>
map Z ZZ
map t <C-t>
map [ <C-]>
nmap <space> zz
map s <C-^>

let mapleader=","
let g:mapleader=","

map <leader>n :cn<cr>
map <leader>p :cp<cr>

" color hack
"hi Normal    guifg=white   guibg=black                    ctermfg=white       ctermbg=black
"hi ErrorMsg  guifg=white   guibg=#287eff                  ctermfg=white       ctermbg=lightblue
"hi Todo      guifg=#d14a14 guibg=#1248d1                  ctermfg=red         ctermbg=darkblue
"hi Search    guifg=#20ffff guibg=#2050d0                  ctermfg=white       ctermbg=darkblue cterm=underline term=underline
"hi IncSearch guifg=#b0ffff guibg=#2050d0                  ctermfg=darkblue    ctermbg=gray

"hi SpecialKey guifg=darkcyan           ctermfg=darkcyan
"hi Directory  guifg=cyan               ctermfg=cyan
"hi Title      guifg=magenta gui=bold   ctermfg=magenta cterm=bold
"hi WarningMsg guifg=red                ctermfg=red
"hi WildMenu   guifg=yellow guibg=black ctermfg=yellow ctermbg=black cterm=none term=none
"hi ModeMsg    guifg=#22cce2            ctermfg=lightblue
"hi MoreMsg    ctermfg=darkgreen        ctermfg=darkgreen
"hi Question   guifg=green gui=none     ctermfg=green cterm=none
"hi NonText    guifg=#0030ff            ctermfg=darkblue

"hi StatusLine   guifg=blue  guibg=darkgray gui=none ctermfg=blue  ctermbg=gray term=none cterm=none
"hi StatusLineNC guifg=black guibg=darkgray gui=none ctermfg=black ctermbg=gray term=none cterm=none
"hi VertSplit    guifg=black guibg=darkgray gui=none ctermfg=black ctermbg=gray term=none cterm=none

"hi Folded     guifg=#808080 guibg=black ctermfg=darkgrey ctermbg=black cterm=bold term=bold
"hi FoldColumn guifg=#808080 guibg=black ctermfg=darkgrey ctermbg=black cterm=bold term=bold
"hi LineNr     guifg=white               ctermfg=green                  cterm=none

"hi DiffAdd    guibg=darkblue                     ctermbg=darkblue cterm=none term=none
"hi DiffChange guibg=darkmagenta                  ctermbg=magenta  cterm=none
"hi DiffDelete guifg=Blue guibg=DarkCyan gui=bold ctermbg=cyan ctermfg=blue
"hi DiffText              guibg=Red      gui=bold ctermbg=red      cterm=bold

"hi Cursor     guifg=yellow guibg=gray  ctermfg=yellow ctermbg=gray
"hi lCursor    guifg=black  guibg=white ctermfg=black ctermbg=white

hi Comment    guifg=#00ff00   ctermfg=lightblue
hi Constant   guifg=#00ff00   ctermfg=green
hi Special    guifg=Orange    ctermfg=green     cterm=bold gui=bold
hi Identifier guifg=#5080ff   ctermfg=blue      cterm=none
hi Statement  guifg=#ffff60   ctermfg=yellow    cterm=bold gui=bold
hi PreProc    guifg=Orange    ctermfg=red       cterm=bold gui=bold
hi type       guifg=#ffff60   ctermfg=cyan      cterm=bold gui=none

"hi Underlined cterm=underline term=underline
"hi Ignore     guifg=bg ctermfg=bg

"hi Pmenu      guifg=#efefef guibg=#333333  ctermfg=black ctermbg=gray
"hi PmenuSel   guifg=#101010 guibg=yellow   ctermfg=black ctermbg=yellow
"hi PmenuSbar  guifg=blue    guibg=darkgray ctermfg=blue  ctermbg=darkgray
"hi PmenuThumb guifg=#c0c0c0
