'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Checkbox } from '@/components/ui/checkbox';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Toggle } from '@/components/ui/toggle';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Calendar } from '@/components/ui/calendar';
import { AspectRatio } from '@/components/ui/aspect-ratio';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from '@/components/ui/breadcrumb';
import { Menubar, MenubarContent, MenubarItem, MenubarMenu, MenubarSeparator, MenubarTrigger } from '@/components/ui/menubar';
import { NavigationMenu, NavigationMenuContent, NavigationMenuItem, NavigationMenuLink, NavigationMenuList, NavigationMenuTrigger } from '@/components/ui/navigation-menu';
import { Pagination, PaginationContent, PaginationItem, PaginationLink, PaginationNext, PaginationPrevious } from '@/components/ui/pagination';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { Drawer, DrawerClose, DrawerContent, DrawerDescription, DrawerFooter, DrawerHeader, DrawerTitle, DrawerTrigger } from '@/components/ui/drawer';
import { toast } from 'sonner';
import { Toaster } from '@/components/ui/sonner';
import { 
  AlertTriangle, 
  Bell, 
  Calendar as CalendarIcon, 
  ChevronDown, 
  ChevronRight,
  Home,
  Info,
  Mail,
  Menu,
  Plus,
  Search,
  Settings,
  Star,
  User,
  X
} from 'lucide-react';

export default function ComponentsPage() {
  const [sliderValue, setSliderValue] = useState([50]);
  const [progressValue, setProgressValue] = useState(33);
  const [switchChecked, setSwitchChecked] = useState(false);
  const [checkboxChecked, setCheckboxChecked] = useState(false);
  const [radioValue, setRadioValue] = useState('option1');
  const [selectValue, setSelectValue] = useState('');
  const [date, setDate] = useState<Date | undefined>(new Date());
  const [isCollapsibleOpen, setIsCollapsibleOpen] = useState(false);

  return (
    <TooltipProvider>
      <div className="container mx-auto py-8 px-4 space-y-8">
        <Toaster />
        
        {/* 页面标题 */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold">shadcn/ui 组件实际使用展示</h1>
          <p className="text-lg text-muted-foreground">
            这里展示了所有 shadcn/ui 组件的实际使用效果，每个组件都是可交互的
          </p>
        </div>

        {/* 基础组件区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">基础组件</Badge>
              按钮、输入框、标签等基础 UI 元素
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 按钮组 */}
            <div className="space-y-2">
              <Label>按钮组件 (Button)</Label>
              <div className="flex gap-2 flex-wrap">
                <Button>默认按钮</Button>
                <Button variant="secondary">次要按钮</Button>
                <Button variant="outline">轮廓按钮</Button>
                <Button variant="ghost">幽灵按钮</Button>
                <Button variant="link">链接按钮</Button>
                <Button variant="destructive">危险按钮</Button>
                <Button size="sm">小按钮</Button>
                <Button size="lg">大按钮</Button>
                <Button disabled>禁用按钮</Button>
              </div>
            </div>

            <Separator />

            {/* 输入框组 */}
            <div className="space-y-4">
              <Label>输入组件</Label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="input1">普通输入框 (Input)</Label>
                  <Input id="input1" placeholder="请输入内容..." />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="input2">密码输入框</Label>
                  <Input id="input2" type="password" placeholder="请输入密码..." />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="textarea1">多行文本框 (Textarea)</Label>
                  <Textarea id="textarea1" placeholder="请输入多行内容..." />
                </div>
              </div>
            </div>

            <Separator />

            {/* 头像和徽章 */}
            <div className="space-y-4">
              <Label>头像和徽章组件</Label>
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <Avatar>
                    <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
                    <AvatarFallback>CN</AvatarFallback>
                  </Avatar>
                  <span>头像组件 (Avatar)</span>
                </div>
                <div className="flex gap-2">
                  <Badge>默认徽章</Badge>
                  <Badge variant="secondary">次要徽章</Badge>
                  <Badge variant="outline">轮廓徽章</Badge>
                  <Badge variant="destructive">危险徽章</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 表单组件区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">表单组件</Badge>
              复选框、单选框、选择器等表单元素
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 复选框和开关 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <Label>复选框 (Checkbox)</Label>
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="checkbox1" 
                    checked={checkboxChecked}
                    onCheckedChange={setCheckboxChecked}
                  />
                  <Label htmlFor="checkbox1">同意服务条款</Label>
                </div>
              </div>
              
              <div className="space-y-4">
                <Label>开关 (Switch)</Label>
                <div className="flex items-center space-x-2">
                  <Switch 
                    id="switch1" 
                    checked={switchChecked}
                    onCheckedChange={setSwitchChecked}
                  />
                  <Label htmlFor="switch1">启用通知</Label>
                </div>
              </div>
            </div>

            <Separator />

            {/* 单选框组 */}
            <div className="space-y-4">
              <Label>单选框组 (RadioGroup)</Label>
              <RadioGroup value={radioValue} onValueChange={setRadioValue}>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="option1" id="option1" />
                  <Label htmlFor="option1">选项 1</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="option2" id="option2" />
                  <Label htmlFor="option2">选项 2</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="option3" id="option3" />
                  <Label htmlFor="option3">选项 3</Label>
                </div>
              </RadioGroup>
            </div>

            <Separator />

            {/* 选择器 */}
            <div className="space-y-4">
              <Label>选择器 (Select)</Label>
              <Select value={selectValue} onValueChange={setSelectValue}>
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="请选择一个选项" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="apple">苹果</SelectItem>
                  <SelectItem value="banana">香蕉</SelectItem>
                  <SelectItem value="orange">橙子</SelectItem>
                  <SelectItem value="grape">葡萄</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Separator />

            {/* 滑块和进度条 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <Label>滑块 (Slider): {sliderValue[0]}</Label>
                <Slider
                  value={sliderValue}
                  onValueChange={setSliderValue}
                  max={100}
                  step={1}
                  className="w-full"
                />
              </div>
              
              <div className="space-y-4">
                <Label>进度条 (Progress): {progressValue}%</Label>
                <Progress value={progressValue} className="w-full" />
                <Button 
                  size="sm" 
                  onClick={() => setProgressValue(Math.min(100, progressValue + 10))}
                >
                  增加进度
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 布局组件区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">布局组件</Badge>
              手风琴、标签页、卡片等布局元素
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 手风琴 */}
            <div className="space-y-4">
              <Label>手风琴 (Accordion)</Label>
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="item-1">
                  <AccordionTrigger>什么是 shadcn/ui？</AccordionTrigger>
                  <AccordionContent>
                    shadcn/ui 是一个基于 Radix UI 和 Tailwind CSS 构建的组件库，提供了美观且可访问的 UI 组件。
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-2">
                  <AccordionTrigger>如何安装组件？</AccordionTrigger>
                  <AccordionContent>
                    使用 npx shadcn@latest add [component-name] 命令来安装特定的组件。
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-3">
                  <AccordionTrigger>支持哪些框架？</AccordionTrigger>
                  <AccordionContent>
                    主要支持 Next.js、Vite、Remix 等现代 React 框架。
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>

            <Separator />

            {/* 标签页 */}
            <div className="space-y-4">
              <Label>标签页 (Tabs)</Label>
              <Tabs defaultValue="tab1" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="tab1">标签 1</TabsTrigger>
                  <TabsTrigger value="tab2">标签 2</TabsTrigger>
                  <TabsTrigger value="tab3">标签 3</TabsTrigger>
                </TabsList>
                <TabsContent value="tab1" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>标签 1 内容</CardTitle>
                      <CardDescription>这是第一个标签页的内容</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p>这里可以放置任何内容，比如表单、图表或其他组件。</p>
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="tab2" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>标签 2 内容</CardTitle>
                      <CardDescription>这是第二个标签页的内容</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p>每个标签页都可以有不同的内容和布局。</p>
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="tab3" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>标签 3 内容</CardTitle>
                      <CardDescription>这是第三个标签页的内容</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p>标签页组件支持键盘导航和无障碍访问。</p>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>

            <Separator />

            {/* 可折叠组件 */}
            <div className="space-y-4">
              <Label>可折叠组件 (Collapsible)</Label>
              <Collapsible open={isCollapsibleOpen} onOpenChange={setIsCollapsibleOpen}>
                <CollapsibleTrigger asChild>
                  <Button variant="outline" className="flex items-center gap-2">
                    {isCollapsibleOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    点击展开/折叠内容
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-4 p-4 border rounded-md">
                  <p>这是可折叠的内容区域。你可以在这里放置任何需要隐藏/显示的内容。</p>
                  <p className="mt-2">可折叠组件常用于 FAQ、设置面板或详细信息展示。</p>
                </CollapsibleContent>
              </Collapsible>
            </div>
          </CardContent>
        </Card>

        {/* 反馈组件区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">反馈组件</Badge>
              警告、对话框、提示等反馈元素
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 警告组件 */}
            <div className="space-y-4">
              <Label>警告组件 (Alert)</Label>
              <div className="space-y-4">
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>信息提示</AlertTitle>
                  <AlertDescription>
                    这是一个信息提示，用于显示一般性的信息内容。
                  </AlertDescription>
                </Alert>
                
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>错误警告</AlertTitle>
                  <AlertDescription>
                    这是一个错误警告，用于显示错误或危险的信息。
                  </AlertDescription>
                </Alert>
              </div>
            </div>

            <Separator />

            {/* 骨架屏 */}
            <div className="space-y-4">
              <Label>骨架屏 (Skeleton)</Label>
              <div className="space-y-2">
                <Skeleton className="h-4 w-[250px]" />
                <Skeleton className="h-4 w-[200px]" />
                <Skeleton className="h-4 w-[150px]" />
              </div>
            </div>

            <Separator />

            {/* 对话框和警告对话框 */}
            <div className="space-y-4">
              <Label>对话框组件</Label>
              <div className="flex gap-4 flex-wrap">
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline">打开对话框</Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>对话框标题</DialogTitle>
                      <DialogDescription>
                        这是一个普通的对话框，可以用来显示详细信息或表单。
                      </DialogDescription>
                    </DialogHeader>
                    <div className="py-4">
                      <p>对话框内容可以是任何 React 组件。</p>
                    </div>
                  </DialogContent>
                </Dialog>

                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="destructive">删除确认</Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>确认删除</AlertDialogTitle>
                      <AlertDialogDescription>
                        此操作无法撤销。这将永久删除数据。
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>取消</AlertDialogCancel>
                      <AlertDialogAction>确认删除</AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>

                <Sheet>
                  <SheetTrigger asChild>
                    <Button variant="outline">打开抽屉</Button>
                  </SheetTrigger>
                  <SheetContent>
                    <SheetHeader>
                      <SheetTitle>抽屉标题</SheetTitle>
                      <SheetDescription>
                        这是一个侧边抽屉组件，常用于设置面板或详细信息展示。
                      </SheetDescription>
                    </SheetHeader>
                    <div className="py-4">
                      <p>抽屉内容区域</p>
                    </div>
                  </SheetContent>
                </Sheet>

                <Drawer>
                  <DrawerTrigger asChild>
                    <Button variant="outline">打开底部抽屉</Button>
                  </DrawerTrigger>
                  <DrawerContent>
                    <DrawerHeader>
                      <DrawerTitle>底部抽屉</DrawerTitle>
                      <DrawerDescription>
                        这是一个从底部滑出的抽屉组件。
                      </DrawerDescription>
                    </DrawerHeader>
                    <div className="p-4">
                      <p>底部抽屉内容</p>
                    </div>
                    <DrawerFooter>
                      <DrawerClose asChild>
                        <Button variant="outline">关闭</Button>
                      </DrawerClose>
                    </DrawerFooter>
                  </DrawerContent>
                </Drawer>

                <Button onClick={() => toast('这是一个 Toast 通知！')}>
                  显示 Toast
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 交互组件区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">交互组件</Badge>
              下拉菜单、工具提示、弹出框等交互元素
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 工具提示 */}
            <div className="space-y-4">
              <Label>工具提示和弹出框</Label>
              <div className="flex gap-4 flex-wrap">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline">悬停显示提示</Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>这是一个工具提示</p>
                  </TooltipContent>
                </Tooltip>

                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline">点击显示弹出框</Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-80">
                    <div className="space-y-2">
                      <h4 className="font-medium leading-none">弹出框标题</h4>
                      <p className="text-sm text-muted-foreground">
                        这是一个弹出框组件，可以包含任何内容。
                      </p>
                    </div>
                  </PopoverContent>
                </Popover>

                <HoverCard>
                  <HoverCardTrigger asChild>
                    <Button variant="link">@shadcn</Button>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80">
                    <div className="flex justify-between space-x-4">
                      <Avatar>
                        <AvatarImage src="https://github.com/shadcn.png" />
                        <AvatarFallback>SC</AvatarFallback>
                      </Avatar>
                      <div className="space-y-1">
                        <h4 className="text-sm font-semibold">@shadcn</h4>
                        <p className="text-sm">
                          shadcn/ui 的创建者，专注于构建美观的用户界面。
                        </p>
                      </div>
                    </div>
                  </HoverCardContent>
                </HoverCard>
              </div>
            </div>

            <Separator />

            {/* 下拉菜单 */}
            <div className="space-y-4">
              <Label>菜单组件</Label>
              <div className="flex gap-4 flex-wrap">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline">
                      下拉菜单 <ChevronDown className="ml-2 h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuLabel>我的账户</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>
                      <User className="mr-2 h-4 w-4" />
                      个人资料
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                      <Settings className="mr-2 h-4 w-4" />
                      设置
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>
                      退出登录
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>

                <ContextMenu>
                  <ContextMenuTrigger asChild>
                    <div className="flex h-[100px] w-[200px] items-center justify-center rounded-md border border-dashed text-sm">
                      右键点击这里
                    </div>
                  </ContextMenuTrigger>
                  <ContextMenuContent>
                    <ContextMenuItem>复制</ContextMenuItem>
                    <ContextMenuItem>粘贴</ContextMenuItem>
                    <ContextMenuItem>删除</ContextMenuItem>
                  </ContextMenuContent>
                </ContextMenu>
              </div>
            </div>

            <Separator />

            {/* 切换按钮 */}
            <div className="space-y-4">
              <Label>切换按钮组件</Label>
              <div className="flex gap-4 flex-wrap items-center">
                <Toggle aria-label="切换斜体">
                  <Star className="h-4 w-4" />
                </Toggle>
                
                <ToggleGroup type="multiple">
                  <ToggleGroupItem value="bold" aria-label="切换粗体">
                    <strong>B</strong>
                  </ToggleGroupItem>
                  <ToggleGroupItem value="italic" aria-label="切换斜体">
                    <em>I</em>
                  </ToggleGroupItem>
                  <ToggleGroupItem value="underline" aria-label="切换下划线">
                    <u>U</u>
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 导航组件区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">导航组件</Badge>
              面包屑、菜单栏、分页等导航元素
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 面包屑 */}
            <div className="space-y-4">
              <Label>面包屑导航 (Breadcrumb)</Label>
              <Breadcrumb>
                <BreadcrumbList>
                  <BreadcrumbItem>
                    <BreadcrumbLink href="/">首页</BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator />
                  <BreadcrumbItem>
                    <BreadcrumbLink href="/components">组件</BreadcrumbLink>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator />
                  <BreadcrumbItem>
                    <BreadcrumbPage>展示页面</BreadcrumbPage>
                  </BreadcrumbItem>
                </BreadcrumbList>
              </Breadcrumb>
            </div>

            <Separator />

            {/* 菜单栏 */}
            <div className="space-y-4">
              <Label>菜单栏 (Menubar)</Label>
              <Menubar>
                <MenubarMenu>
                  <MenubarTrigger>文件</MenubarTrigger>
                  <MenubarContent>
                    <MenubarItem>新建</MenubarItem>
                    <MenubarItem>打开</MenubarItem>
                    <MenubarSeparator />
                    <MenubarItem>保存</MenubarItem>
                  </MenubarContent>
                </MenubarMenu>
                <MenubarMenu>
                  <MenubarTrigger>编辑</MenubarTrigger>
                  <MenubarContent>
                    <MenubarItem>撤销</MenubarItem>
                    <MenubarItem>重做</MenubarItem>
                    <MenubarSeparator />
                    <MenubarItem>复制</MenubarItem>
                    <MenubarItem>粘贴</MenubarItem>
                  </MenubarContent>
                </MenubarMenu>
                <MenubarMenu>
                  <MenubarTrigger>查看</MenubarTrigger>
                  <MenubarContent>
                    <MenubarItem>缩放</MenubarItem>
                    <MenubarItem>全屏</MenubarItem>
                  </MenubarContent>
                </MenubarMenu>
              </Menubar>
            </div>

            <Separator />

            {/* 分页 */}
            <div className="space-y-4">
              <Label>分页组件 (Pagination)</Label>
              <Pagination>
                <PaginationContent>
                  <PaginationItem>
                    <PaginationPrevious href="#" />
                  </PaginationItem>
                  <PaginationItem>
                    <PaginationLink href="#">1</PaginationLink>
                  </PaginationItem>
                  <PaginationItem>
                    <PaginationLink href="#" isActive>
                      2
                    </PaginationLink>
                  </PaginationItem>
                  <PaginationItem>
                    <PaginationLink href="#">3</PaginationLink>
                  </PaginationItem>
                  <PaginationItem>
                    <PaginationNext href="#" />
                  </PaginationItem>
                </PaginationContent>
              </Pagination>
            </div>
          </CardContent>
        </Card>

        {/* 数据展示区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">数据展示</Badge>
              表格、日历等数据展示元素
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 表格 */}
            <div className="space-y-4">
              <Label>表格组件 (Table)</Label>
              <Table>
                <TableCaption>最近的订单列表</TableCaption>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">订单号</TableHead>
                    <TableHead>状态</TableHead>
                    <TableHead>方法</TableHead>
                    <TableHead className="text-right">金额</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell className="font-medium">INV001</TableCell>
                    <TableCell>已支付</TableCell>
                    <TableCell>信用卡</TableCell>
                    <TableCell className="text-right">¥250.00</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">INV002</TableCell>
                    <TableCell>待支付</TableCell>
                    <TableCell>PayPal</TableCell>
                    <TableCell className="text-right">¥150.00</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">INV003</TableCell>
                    <TableCell>未支付</TableCell>
                    <TableCell>银行转账</TableCell>
                    <TableCell className="text-right">¥350.00</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </div>

            <Separator />

            {/* 日历 */}
            <div className="space-y-4">
              <Label>日历组件 (Calendar)</Label>
              <div className="flex justify-center">
                <Calendar
                  mode="single"
                  selected={date}
                  onSelect={setDate}
                  className="rounded-md border"
                />
              </div>
            </div>

            <Separator />

            {/* 宽高比和滚动区域 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <Label>宽高比组件 (AspectRatio)</Label>
                <AspectRatio ratio={16 / 9} className="bg-muted">
                  <div className="flex items-center justify-center h-full">
                    <p className="text-sm text-muted-foreground">16:9 宽高比</p>
                  </div>
                </AspectRatio>
              </div>
              
              <div className="space-y-4">
                <Label>滚动区域 (ScrollArea)</Label>
                <ScrollArea className="h-[200px] w-full rounded-md border p-4">
                  <div className="space-y-2">
                    {Array.from({ length: 20 }, (_, i) => (
                      <div key={i} className="text-sm">
                        滚动内容项目 {i + 1}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 其他组件区域 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant="secondary">其他组件</Badge>
              可调整大小面板、命令面板等特殊组件
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 可调整大小的面板 */}
            <div className="space-y-4">
              <Label>可调整大小的面板 (Resizable)</Label>
              <ResizablePanelGroup direction="horizontal" className="min-h-[200px] rounded-lg border">
                <ResizablePanel defaultSize={50}>
                  <div className="flex h-full items-center justify-center p-6">
                    <span className="font-semibold">左侧面板</span>
                  </div>
                </ResizablePanel>
                <ResizableHandle withHandle />
                <ResizablePanel defaultSize={50}>
                  <div className="flex h-full items-center justify-center p-6">
                    <span className="font-semibold">右侧面板</span>
                  </div>
                </ResizablePanel>
              </ResizablePanelGroup>
            </div>

            <Separator />

            {/* 命令面板 */}
            <div className="space-y-4">
              <Label>命令面板 (Command)</Label>
              <Command className="rounded-lg border shadow-md">
                <CommandInput placeholder="搜索命令..." />
                <CommandList>
                  <CommandEmpty>没有找到结果。</CommandEmpty>
                  <CommandGroup heading="建议">
                    <CommandItem>
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      <span>日历</span>
                    </CommandItem>
                    <CommandItem>
                      <Search className="mr-2 h-4 w-4" />
                      <span>搜索表情符号</span>
                    </CommandItem>
                    <CommandItem>
                      <Settings className="mr-2 h-4 w-4" />
                      <span>设置</span>
                    </CommandItem>
                  </CommandGroup>
                </CommandList>
              </Command>
            </div>
          </CardContent>
        </Card>

        {/* 页面底部 */}
        <div className="text-center py-8">
          <p className="text-muted-foreground">
            以上展示了 shadcn/ui 组件库中的主要组件，每个组件都是完全可交互的。
          </p>
          <p className="text-muted-foreground mt-2">
            更多组件和详细文档请访问 <a href="https://ui.shadcn.com" className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">官方网站</a>
          </p>
        </div>
      </div>
    </TooltipProvider>
  );
}